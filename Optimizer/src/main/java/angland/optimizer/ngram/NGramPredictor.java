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
import angland.optimizer.var.matrix.IMatrixValue;
import angland.optimizer.var.scalar.IScalarValue;
import angland.optimizer.var.scalar.MappedDerivativeScalar;
import angland.optimizer.var.scalar.StreamingSum;

public class NGramPredictor {

  private final Context<String> context;
  private final IMatrixValue<String> embedding;
  private final IMatrixValue<String> responseBias;
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
        IMatrixValue.varOrConst("embedding", cellTemplate.getSize(), vocabulary, context, constant);
    this.responseBias = IMatrixValue.varOrConst("responseBias", 1, vocabulary, context, constant);
    this.cell = cellTemplate.create(context);
    this.context = context;
  }

  public Context<String> getContext() {
    return context;
  }


  public List<Integer> predictNext(List<Integer> inputInts, int predictTokens, int unkIdx) {
    IMatrixValue<String> hiddenState =
        IMatrixValue.repeat(IScalarValue.constant(0), cell.getSize(), 1);
    IMatrixValue<String> lastOutput =
        IMatrixValue.repeat(IScalarValue.constant(0), cell.getSize(), 1);
    for (int i : inputInts) {
      IMatrixValue<String> selectedCol = embedding.getColumn(IScalarValue.constant(i));

      RnnStateTuple<String> inputState = new RnnStateTuple<String>(hiddenState, selectedCol);
      RnnStateTuple<String> outputState = cell.apply(inputState).toConstant();
      hiddenState = outputState.getHiddenState();
      lastOutput = outputState.getExposedState();
    }
    ArrayMatrixValue.Builder<String> unkRemoverBuilder =
        new ArrayMatrixValue.Builder<>(embedding.getWidth(), 1);
    for (int i = 0; i < embedding.getWidth(); ++i) {
      unkRemoverBuilder
          .set(i, 0, i == unkIdx ? IScalarValue.constant(0) : IScalarValue.constant(1));
    }
    ArrayMatrixValue<String> unkRemover = unkRemoverBuilder.build();
    List<Integer> outputs = new ArrayList<>();
    Consumer<IMatrixValue<String>> addOutput =
        state -> {
          IMatrixValue<String> tokenActivation =
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

  public IScalarValue<String> getLoss(List<Integer> inputInts, int samples) {
    if (inputInts.size() < 2) {
      throw new IllegalArgumentException("Can only compute loss on at least two elements.");
    }
    List<IScalarValue<String>> lossComponents = new ArrayList<>();
    IMatrixValue<String> hiddenState =
        IMatrixValue.repeat(IScalarValue.constant(0), cell.getSize(), 1);
    for (int i = 0; i < inputInts.size() - 1; ++i) {
      int input = inputInts.get(i);
      int output = inputInts.get(i + 1);
      IMatrixValue<String> selectedCol = embedding.getColumn(IScalarValue.constant(input));
      RnnStateTuple<String> inputState = new RnnStateTuple<>(hiddenState, selectedCol);
      RnnStateTuple<String> outputState = cell.apply(inputState);
      hiddenState = outputState.getHiddenState();
      List<Integer> selectedIndices =
          IMatrixValue.selectAndSample(embedding.getWidth(), samples, output);
      IMatrixValue<String> sampledEmbedding = embedding.getColumns(selectedIndices).transpose();
      IMatrixValue<String> sampledBias = responseBias.getColumns(selectedIndices);
      IMatrixValue<String> softmaxInput =
          sampledEmbedding.streamingTimes(
              outputState.getExposedState().transform(IScalarValue::cache)).plus(
              sampledBias.transpose());
      IScalarValue<String> max = softmaxInput.get(0, 0);
      for (int j = 1; j < softmaxInput.getHeight(); ++j) {
        if (softmaxInput.get(j, 0).value() > max.value()) {
          max = softmaxInput.get(j, 0);
        }
      }
      IScalarValue<String> maxConstant = max.toConstant();
      softmaxInput = softmaxInput.transform(x -> x.minus(maxConstant).exp());
      IScalarValue<String> softmaxNum = softmaxInput.get(0, 0);
      IScalarValue<String> softmaxDenom = softmaxInput.elementSumStream().arrayCache(context);
      IScalarValue<String> epsilon = IScalarValue.constant(.0001);
      IScalarValue<String> softmaxOfCorrectIdx =
          softmaxNum.divide(softmaxDenom).plus(epsilon).ln().cache();
      lossComponents.add(softmaxOfCorrectIdx);
    }
    return new StreamingSum<>(lossComponents).arrayCache(context).times(IScalarValue.constant(-1))
        .divide(IScalarValue.constant(inputInts.size() - 1));
  }

  public IScalarValue<String> getBatchLoss(Collection<List<Integer>> inputs, ExecutorService es,
      int samples) {
    List<Callable<IScalarValue<String>>> losses = new ArrayList<>();
    inputs.forEach(input -> losses.add(() -> getLoss(input, samples)));
    try {

      MappedDerivativeScalar.Builder<String> resultBuilder =
          new MappedDerivativeScalar.Builder<>(cell.getSize() + embedding.getWidth()
              * embedding.getHeight());
      List<Callable<IScalarValue<String>>> tasks = new ArrayList<>();
      inputs.forEach(x -> tasks.add(() -> getLoss(x, samples)));
      es.invokeAll(tasks).forEach(loss -> {
        try {
          resultBuilder.increment(loss.get());
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      });
      return resultBuilder.build().divide(IScalarValue.constant(inputs.size()));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

  }
}
