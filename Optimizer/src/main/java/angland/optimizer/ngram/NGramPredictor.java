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

import angland.optimizer.nn.LstmCell;
import angland.optimizer.nn.LstmStateTuple;
import angland.optimizer.var.Context;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.matrix.IMatrixValue;
import angland.optimizer.var.scalar.IScalarValue;
import angland.optimizer.var.scalar.MappedDerivativeScalar;

public class NGramPredictor {

  private final Context<String> context;
  private final IMatrixValue<String> embedding;
  private final IMatrixValue<String> responseBias;
  private final LstmCell cell;


  public static Map<IndexedKey<String>, Double> randomizedContext(int vocabulary, int lstmSize) {
    Map<IndexedKey<String>, Double> map = new HashMap<>();
    getKeys(vocabulary, lstmSize).forEach(k -> map.put(k, Math.random() * 2 - 1));
    return map;
  }

  public static Stream<IndexedKey<String>> getKeys(int vocabulary, int lstmSize) {
    return Stream.concat(
        IndexedKey.getAllMatrixKeys("embedding", lstmSize, vocabulary).stream(),
        Stream.concat(LstmCell.getKeys("cell", lstmSize),
            IndexedKey.getAllMatrixKeys("responseBias", 1, vocabulary).stream()));
  }

  public NGramPredictor(int vocabulary, int lstmSize, Context<String> context,
      double gradientClipThreshold, boolean constant) {
    this.embedding = IMatrixValue.varOrConst("embedding", lstmSize, vocabulary, context, constant);
    this.responseBias = IMatrixValue.varOrConst("responseBias", 1, vocabulary, context, constant);
    this.cell = new LstmCell("cell", lstmSize, context, gradientClipThreshold, constant);
    this.context = context;
  }

  public Context<String> getContext() {
    return context;
  }


  public List<Integer> predictNext(List<Integer> inputInts, int predictTokens) {
    IMatrixValue<String> hiddenState =
        IMatrixValue.repeat(IScalarValue.constant(0), cell.getSize(), 1);
    IMatrixValue<String> lastOutput =
        IMatrixValue.repeat(IScalarValue.constant(0), cell.getSize(), 1);
    for (int i : inputInts) {
      IMatrixValue<String> selectedCol = embedding.getColumn(IScalarValue.constant(i));

      LstmStateTuple<String> inputState = new LstmStateTuple<String>(hiddenState, selectedCol);
      LstmStateTuple<String> outputState = cell.apply(inputState).toConstant();
      hiddenState = outputState.getHiddenState();
      lastOutput = outputState.getExposedState();
    }
    List<Integer> outputs = new ArrayList<>();
    Consumer<IMatrixValue<String>> addOutput =
        state -> outputs.add((int) embedding.transpose().times(state).softmax().maxIdx().value());
    addOutput.accept(lastOutput);
    LstmStateTuple<String> lastState = new LstmStateTuple<>(hiddenState, lastOutput);
    for (int i = 0; i < predictTokens; ++i) {
      LstmStateTuple<String> nextState = cell.apply(lastState);
      lastState = nextState.toConstant();
      addOutput.accept(lastState.getExposedState());
    }

    return outputs;
  }


  public IScalarValue<String> getLoss(List<Integer> inputInts, int samples) {
    if (inputInts.size() < 2) {
      throw new IllegalArgumentException("Can only compute loss on at least two elements.");
    }
    MappedDerivativeScalar.Builder<String> lossBuilder =
        new MappedDerivativeScalar.Builder<>(cell.getSize());
    IMatrixValue<String> hiddenState =
        IMatrixValue.repeat(IScalarValue.constant(0), cell.getSize(), 1);
    for (int i = 0; i < inputInts.size() - 1; ++i) {
      int input = inputInts.get(i);
      int output = inputInts.get(i + 1);
      IMatrixValue<String> selectedCol = embedding.getColumn(IScalarValue.constant(input));
      LstmStateTuple<String> inputState = new LstmStateTuple<>(hiddenState, selectedCol);
      LstmStateTuple<String> outputState = cell.apply(inputState);
      hiddenState = outputState.getHiddenState();
      List<Integer> selectedIndices =
          IMatrixValue.selectAndSample(embedding.getWidth(), samples, output);
      IMatrixValue<String> sampledEmbedding = embedding.getColumns(selectedIndices).transpose();
      IMatrixValue<String> sampledBias = responseBias.getColumns(selectedIndices);
      IMatrixValue<String> softmaxInput =
          sampledEmbedding
              .streamingTimes(outputState.getExposedState().transform(IScalarValue::cache))
              .plus(sampledBias.transpose()).transform(IScalarValue::exp);
      IScalarValue<String> softmaxNum = softmaxInput.get(0, 0);
      IScalarValue<String> softmaxDenom = softmaxInput.elementSumStream().arrayCache(context);
      IScalarValue<String> softmaxOfCorrectIdx = softmaxNum.divide(softmaxDenom).ln();
      lossBuilder.increment(softmaxOfCorrectIdx);

    }
    return lossBuilder.build().times(IScalarValue.constant(-1))
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
