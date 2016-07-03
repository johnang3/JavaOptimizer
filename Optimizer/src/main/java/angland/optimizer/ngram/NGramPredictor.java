package angland.optimizer.ngram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Stream;

import angland.optimizer.nn.LstmCell;
import angland.optimizer.nn.LstmStateTupleExpression;
import angland.optimizer.nn.LstmStateTupleValue;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.MatrixExpression;
import angland.optimizer.var.ScalarExpression;

public class NGramPredictor {

  private Map<IndexedKey<String>, Double> context;
  private final MatrixExpression<String> embedding;
  private final LstmCell cell;

  public NGramPredictor(int vocabulary, int lstmSize) {
    this.embedding = MatrixExpression.variable("embedding", lstmSize, vocabulary);
    this.cell = new LstmCell("cell", lstmSize);
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    Stream.concat(IndexedKey.getAllMatrixKeys("embedding", cell.getSize(), vocabulary).stream(),
        cell.getKeys()).forEach(k -> {
      context.put(k, Math.random() * 2 - 1);
    });
    this.context = context;
  }


  public Map<IndexedKey<String>, Double> getContext() {
    return context;
  }

  public void setContext(Map<IndexedKey<String>, Double> context) {
    this.context = context;
  }


  public List<Integer> predictNext(List<Integer> inputInts, int predictTokens) {
    MatrixExpression<String> hiddenState =
        MatrixExpression.repeat(ScalarExpression.constant(0), cell.getSize(), 1);
    MatrixExpression<String> lastOutput =
        MatrixExpression.repeat(ScalarExpression.constant(0), cell.getSize(), 1);
    for (int i : inputInts) {
      MatrixExpression<String> selectedCol = embedding.getColumn(ScalarExpression.constant(i));

      LstmStateTupleExpression<String> inputState =
          new LstmStateTupleExpression<String>(hiddenState, selectedCol);
      LstmStateTupleExpression<String> outputState = cell.apply(inputState);
      LstmStateTupleValue<String> outputStateValue = outputState.evaluate(context);
      LstmStateTupleExpression<String> outputStateAsConstant = outputStateValue.toConstant();
      hiddenState = outputStateAsConstant.getHiddenState();
      lastOutput = outputStateAsConstant.getExposedState();
    }
    List<Integer> outputs = new ArrayList<>();
    Consumer<MatrixExpression<String>> addOutput =
        state -> outputs.add((int) embedding.columnProximity(state).transform(v -> v.power(-1))
            .transpose().softmax().maxIdx().evaluate(context).value());
    addOutput.accept(lastOutput);
    LstmStateTupleExpression<String> lastState =
        new LstmStateTupleExpression<>(hiddenState, lastOutput);
    for (int i = 0; i < predictTokens; ++i) {
      LstmStateTupleExpression<String> nextState = cell.apply(lastState);
      LstmStateTupleValue<String> nextStateValue = nextState.evaluate(context);
      lastState = nextStateValue.toConstant();
      addOutput.accept(lastState.getExposedState());
    }

    return outputs;
  }
}
