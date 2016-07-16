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
import angland.optimizer.var.IMatrixValue;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.scalar.IScalarValue;
import angland.optimizer.var.scalar.MappedDerivativeScalar;

public class NGramPredictor {

  private final Map<IndexedKey<String>, Double> context;
  private final IMatrixValue<String> embedding;
  private final LstmCell cell;

  public static Map<IndexedKey<String>, Double> randomizedContext(int vocabulary, int lstmSize) {
    Map<IndexedKey<String>, Double> map = new HashMap<>();
    getKeys(vocabulary, lstmSize).forEach(k -> map.put(k, Math.random() * 2 - 1));
    return map;
  }

  public static Stream<IndexedKey<String>> getKeys(int vocabulary, int lstmSize) {
    return Stream.concat(IndexedKey.getAllMatrixKeys("embedding", lstmSize, vocabulary).stream(),
        LstmCell.getKeys("cell", lstmSize));
  }

  public NGramPredictor(int vocabulary, int lstmSize, Map<IndexedKey<String>, Double> context,
      double gradientClipThreshold) {
    this.embedding = IMatrixValue.var("embedding", lstmSize, vocabulary, context);
    this.cell = new LstmCell("cell", lstmSize, context, gradientClipThreshold);
    this.context = context;
  }

  public Map<IndexedKey<String>, Double> getContext() {
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

  private static final IMatrixValue<String> val = IMatrixValue.repeat(IScalarValue.constant(0),
      200, 1);

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
      IMatrixValue<String> sampledEmbedding =
          embedding.selectAndSampleColumnsWithElimination(output, samples).transpose();
      IMatrixValue<String> softmaxInput =
          sampledEmbedding.times(outputState.getExposedState().transform(IScalarValue::cache))
              .transform(IScalarValue::exp);
      lossBuilder.increment(softmaxInput.get(0, 0).divide(softmaxInput.elementSum()).ln());

    }
    return lossBuilder.build().times(IScalarValue.constant(-1))
        .divide(IScalarValue.constant(inputInts.size() - 1));
  }

  public MappedDerivativeScalar<String> getBatchLoss(Collection<List<Integer>> inputs,
      ExecutorService es, int samples) {
    List<Callable<IScalarValue<String>>> losses = new ArrayList<>();
    inputs.forEach(input -> losses.add(() -> getLoss(input, samples)));
    try {

      MappedDerivativeScalar.Builder<String> resultBuilder =
          new MappedDerivativeScalar.Builder<>(cell.getSize() + embedding.getWidth()
              * embedding.getHeight());
      inputs.forEach(input -> resultBuilder.increment(getLoss(input, samples)));
      return resultBuilder.build();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

  }
}
