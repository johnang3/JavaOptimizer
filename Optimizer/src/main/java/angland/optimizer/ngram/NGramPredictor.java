package angland.optimizer.ngram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Stream;

import angland.optimizer.nn.LstmCell;
import angland.optimizer.nn.LstmStateTuple;
import angland.optimizer.var.ArrayMatrixValue;
import angland.optimizer.var.IMatrixValue;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.ScalarValue;

public class NGramPredictor {

  private Map<IndexedKey<String>, Double> context;
  private IMatrixValue<String> embedding;
  private LstmCell cell;

  public NGramPredictor(int vocabulary, int lstmSize) {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    Stream.concat(IndexedKey.getAllMatrixKeys("embedding", lstmSize, vocabulary).stream(),
        LstmCell.getKeys("cell", lstmSize)).forEach(k -> {
      context.put(k, Math.random() * 2 - 1);
    });
    this.embedding = IMatrixValue.var("embedding", lstmSize, vocabulary, context);
    this.cell = new LstmCell("cell", lstmSize, context);

    this.context = context;
  }


  public Map<IndexedKey<String>, Double> getContext() {
    return context;
  }

  public void setContext(Map<IndexedKey<String>, Double> context) {
    this.context = context;
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
          new ArrayMatrixValue.Builder<>(cell.getSize(), 1);
      for (int j = 0; j < cell.getSize(); ++j) {
        targetBuilder.set(j, 0, ScalarValue.constant(j == output ? 1 : 0));
      }
      IMatrixValue<String> target = targetBuilder.build();
      ScalarValue<String> epsilon = ScalarValue.constant(.0001);
      IMatrixValue<String> crossEntropy =
          softmax.pointwise(target, (a, b) -> a.plus(epsilon).ln().times(b));
      lossBuilder.increment(crossEntropy.elementSum());
    }
    return lossBuilder.build();
  }
}
