package angland.optimizer.ngram;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.stream.Stream;

import angland.optimizer.nn.LstmCell;
import angland.optimizer.nn.LstmStateTuple;
import angland.optimizer.var.ArrayMatrixValue;
import angland.optimizer.var.IMatrixValue;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.ScalarValue;

public class NGramPredictor {

  private final Map<IndexedKey<String>, Double> context;
  private final IMatrixValue<String> embedding;
  private final LstmCell cell;
  private final ScalarValue<String> one = ScalarValue.constant(1);

  public static Map<IndexedKey<String>, Double> randomizedContext(int vocabulary, int lstmSize) {
    Map<IndexedKey<String>, Double> map = new HashMap<>();
    getKeys(vocabulary, lstmSize).forEach(k -> map.put(k, Math.random() * 2 - 1));
    return map;
  }

  public static Stream<IndexedKey<String>> getKeys(int vocabulary, int lstmSize) {
    return Stream.concat(IndexedKey.getAllMatrixKeys("embedding", lstmSize, vocabulary).stream(),
        LstmCell.getKeys("cell", lstmSize));
  }

  public NGramPredictor(int vocabulary, int lstmSize, Map<IndexedKey<String>, Double> context) {
    this.embedding = IMatrixValue.var("embedding", lstmSize, vocabulary, context);
    this.cell = new LstmCell("cell", lstmSize, context);
    this.context = context;
  }

  public Map<IndexedKey<String>, Double> getContext() {
    return context;
  }


  public List<Integer> predictNext(List<Integer> inputInts, int predictTokens) {
    IMatrixValue<String> hiddenState =
        IMatrixValue.repeat(ScalarValue.constant(0), cell.getSize(), 1);
    IMatrixValue<String> lastOutput =
        IMatrixValue.repeat(ScalarValue.constant(0), cell.getSize(), 1);
    for (int i : inputInts) {
      IMatrixValue<String> selectedCol = embedding.getColumn(ScalarValue.constant(i));

      LstmStateTuple<String> inputState = new LstmStateTuple<String>(hiddenState, selectedCol);
      LstmStateTuple<String> outputState = cell.apply(inputState).toConstant();
      hiddenState = outputState.getHiddenState();
      lastOutput = outputState.getExposedState();
    }
    List<Integer> outputs = new ArrayList<>();
    Consumer<IMatrixValue<String>> addOutput =
        state -> outputs.add((int) embedding.columnProximity(state).transform(v -> v.power(-1))
            .transpose().softmax().maxIdx().value());
    addOutput.accept(lastOutput);
    LstmStateTuple<String> lastState = new LstmStateTuple<>(hiddenState, lastOutput);
    for (int i = 0; i < predictTokens; ++i) {
      LstmStateTuple<String> nextState = cell.apply(lastState);
      lastState = nextState.toConstant();
      addOutput.accept(lastState.getExposedState());
    }

    return outputs;
  }

  public ScalarValue<String> getLoss(List<Integer> inputInts) {
    if (inputInts.size() < 2) {
      throw new IllegalArgumentException("Can only compute loss on at least two elements.");
    }
    ScalarValue.Builder<String> lossBuilder = new ScalarValue.Builder<>(cell.getSize());
    IMatrixValue<String> hiddenState =
        IMatrixValue.repeat(ScalarValue.constant(0), cell.getSize(), 1);
    for (int i = 0; i < inputInts.size() - 1; ++i) {
      int input = inputInts.get(i);
      int output = inputInts.get(i + 1);
      IMatrixValue<String> selectedCol = embedding.getColumn(ScalarValue.constant(input));
      LstmStateTuple<String> inputState = new LstmStateTuple<>(hiddenState, selectedCol);
      LstmStateTuple<String> outputState = cell.apply(inputState);
      hiddenState = outputState.getHiddenState();
      IMatrixValue<String> softmax =
          embedding.columnProximity(outputState.getExposedState()).transform(v -> v.power(-1))
              .transpose().softmax();
      ArrayMatrixValue.Builder<String> targetBuilder =
          new ArrayMatrixValue.Builder<>(embedding.getWidth(), 1);
      for (int j = 0; j < embedding.getWidth(); ++j) {
        targetBuilder.set(j, 0, ScalarValue.constant(j == output ? 1 : 0));
      }
      IMatrixValue<String> target = targetBuilder.build();
      ScalarValue<String> epsilon = ScalarValue.constant(.0001);
      BinaryOperator<ScalarValue<String>> ce = (a, b) -> a.plus(epsilon).ln().times(b);
      BinaryOperator<ScalarValue<String>> twoSidedCe =
          (a, b) -> ce.apply(a, b).plus(ce.apply(one.minus(a), one.minus(b)));
      IMatrixValue<String> crossEntropy = softmax.pointwise(target, (a, b) -> b.times(a.ln()));
      /*
       * StringBuilder sb = new StringBuilder("Target: "); StringBuilder sb2 = new
       * StringBuilder("Softmax: "); StringBuilder sb3 = new StringBuilder("CrossEntropy: "); for
       * (int j = 0; j < target.getHeight(); ++j) { sb.append(target.get(j, 0).value() + ", ");
       * sb2.append(softmax.get(j, 0).value() + ", "); sb3.append(crossEntropy.get(j, 0).value() +
       * ", "); } System.out.println(sb.toString()); System.out.println(sb2.toString());
       * System.out.println(sb3.toString());
       */
      lossBuilder.increment(crossEntropy.elementSum().divide(
          ScalarValue.constant(target.getHeight())));
    }
    return lossBuilder.build().times(ScalarValue.constant(-1))
        .divide(ScalarValue.constant(inputInts.size() - 1));
  }

  public ScalarValue<String> getBatchLoss(Collection<List<Integer>> inputs, ExecutorService es) {
    List<Callable<ScalarValue<String>>> losses = new ArrayList<>();
    inputs.forEach(input -> losses.add(() -> getLoss(input)));
    try {

      ScalarValue.Builder<String> resultBuilder =
          new ScalarValue.Builder<>(cell.getSize() + embedding.getWidth() * embedding.getHeight());
      inputs.forEach(input -> resultBuilder.increment(getLoss(input)));
      return resultBuilder.build();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

  }
}
