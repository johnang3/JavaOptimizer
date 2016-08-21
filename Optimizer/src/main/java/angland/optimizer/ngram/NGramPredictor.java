package angland.optimizer.ngram;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;
import java.util.stream.Stream;

import angland.optimizer.nn.RnnCell;
import angland.optimizer.nn.RnnCellTemplate;
import angland.optimizer.nn.RnnStateTuple;
import angland.optimizer.var.Context;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.matrix.ArrayMatrixValue;
import angland.optimizer.var.matrix.Matrix;
import angland.optimizer.var.scalar.Scalar;
import angland.optimizer.var.scalar.MappedDerivativeScalar;
import angland.optimizer.var.scalar.StreamingSum;

public class NGramPredictor {

  private final Context<String> context;
  private final Matrix<String> embedding;
  private final Matrix<String> responseBias;
  private final RnnCell<String> cell;


  public static Map<IndexedKey<String>, Double> randomizedContext(int vocabulary,
      RnnCellTemplate cellTemplate) {
    Map<IndexedKey<String>, Double> map = new HashMap<>();
    getKeys(vocabulary, cellTemplate).forEach(k -> map.put(k, Math.random() * 2 - 1));
    return map;
  }

  public static Stream<IndexedKey<String>> getKeys(int vocabulary, RnnCellTemplate cellTemplate) {
    return Stream.concat(
        IndexedKey.getAllMatrixKeys("embedding", cellTemplate.getSize(), vocabulary).stream(),
        Stream.concat(cellTemplate.getKeys(),
            IndexedKey.getAllMatrixKeys("responseBias", 1, vocabulary).stream()));
  }

  public NGramPredictor(int vocabulary, RnnCellTemplate cellTemplate, Context<String> context,
      boolean constant) {
    this.embedding =
        Matrix.varOrConst("embedding", cellTemplate.getSize(), vocabulary, context, constant);
    this.responseBias = Matrix.varOrConst("responseBias", 1, vocabulary, context, constant);
    this.cell = cellTemplate.create(context);
    this.context = context;
  }

  public Context<String> getContext() {
    return context;
  }


  public List<Integer> predictNext(List<Integer> inputInts, int predictTokens, int unkIdx) {
    Matrix<String> hiddenState =
        Matrix.repeat(Scalar.constant(0), cell.getSize(), 1);
    Matrix<String> lastOutput =
        Matrix.repeat(Scalar.constant(0), cell.getSize(), 1);
    for (int i : inputInts) {
      Matrix<String> selectedCol = embedding.getColumn(Scalar.constant(i));

      RnnStateTuple<String> inputState = new RnnStateTuple<String>(hiddenState, selectedCol);
      RnnStateTuple<String> outputState = cell.apply(inputState).toConstant();
      hiddenState = outputState.getHiddenState();
      lastOutput = outputState.getExposedState();
    }
    ArrayMatrixValue.Builder<String> unkRemoverBuilder =
        new ArrayMatrixValue.Builder<>(embedding.getWidth(), 1);
    for (int i = 0; i < embedding.getWidth(); ++i) {
      unkRemoverBuilder
          .set(i, 0, i == unkIdx ? Scalar.constant(0) : Scalar.constant(1));
    }
    ArrayMatrixValue<String> unkRemover = unkRemoverBuilder.build();
    List<Integer> outputs = new ArrayList<>();
    Consumer<Matrix<String>> addOutput =
        state -> {
          Matrix<String> tokenActivation =
              embedding.transpose().times(state).pointwiseMultiply(unkRemover);
          outputs.add((int) tokenActivation.softmax().maxIdx().value());
        };
    addOutput.accept(lastOutput);
    RnnStateTuple<String> lastState = new RnnStateTuple<>(hiddenState, lastOutput);
    for (int i = 0; i < predictTokens; ++i) {
      RnnStateTuple<String> nextState = cell.apply(lastState);
      lastState = nextState.toConstant();
      addOutput.accept(lastState.getExposedState());
    }

    return outputs;
  }

  public Scalar<String> getLoss(List<Integer> inputInts, int samples) {
    if (inputInts.size() < 2) {
      throw new IllegalArgumentException("Can only compute loss on at least two elements.");
    }
    List<Scalar<String>> lossComponents = new ArrayList<>();
    Matrix<String> hiddenState =
        Matrix.repeat(Scalar.constant(0), cell.getSize(), 1);
    for (int i = 0; i < inputInts.size() - 1; ++i) {
      int input = inputInts.get(i);
      int output = inputInts.get(i + 1);
      Matrix<String> selectedCol = embedding.getColumn(Scalar.constant(input));
      RnnStateTuple<String> inputState = new RnnStateTuple<>(hiddenState, selectedCol);
      RnnStateTuple<String> outputState = cell.apply(inputState);
      hiddenState = outputState.getHiddenState();
      List<Integer> selectedIndices =
          Matrix.selectAndSample(embedding.getWidth(), samples, output);
      Matrix<String> sampledEmbedding = embedding.getColumns(selectedIndices).transpose();
      Matrix<String> sampledBias = responseBias.getColumns(selectedIndices);
      Matrix<String> softmaxInput =
          sampledEmbedding.streamingTimes(
              outputState.getExposedState().transform(Scalar::cache)).plus(
              sampledBias.transpose());
      Scalar<String> max = softmaxInput.get(0, 0);
      for (int j = 1; j < softmaxInput.getHeight(); ++j) {
        if (softmaxInput.get(j, 0).value() > max.value()) {
          max = softmaxInput.get(j, 0);
        }
      }
      Scalar<String> maxConstant = max.toConstant();
      softmaxInput = softmaxInput.transform(x -> x.minus(maxConstant).exp());
      Scalar<String> softmaxNum = softmaxInput.get(0, 0);
      Scalar<String> softmaxDenom = softmaxInput.elementSumStream().arrayCache(context);
      Scalar<String> epsilon = Scalar.constant(.0001);
      Scalar<String> softmaxOfCorrectIdx =
          softmaxNum.divide(softmaxDenom).plus(epsilon).ln().cache();
      lossComponents.add(softmaxOfCorrectIdx);
    }
    return new StreamingSum<>(lossComponents).arrayCache(context).times(Scalar.constant(-1))
        .divide(Scalar.constant(inputInts.size() - 1));
  }

  public Scalar<String> getBatchLoss(Collection<List<Integer>> inputs, ExecutorService es,
      int samples) {
    List<Callable<Scalar<String>>> losses = new ArrayList<>();
    inputs.forEach(input -> losses.add(() -> getLoss(input, samples)));
    try {

      MappedDerivativeScalar.Builder<String> resultBuilder =
          new MappedDerivativeScalar.Builder<>(cell.getSize() + embedding.getWidth()
              * embedding.getHeight());
      List<Callable<Scalar<String>>> tasks = new ArrayList<>();
      inputs.forEach(x -> tasks.add(() -> getLoss(x, samples)));
      es.invokeAll(tasks).forEach(loss -> {
        try {
          resultBuilder.increment(loss.get());
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      });
      return resultBuilder.build().divide(Scalar.constant(inputs.size()));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

  }
}
