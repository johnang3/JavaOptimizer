package angland.optimizer.ngram;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import angland.optimizer.nn.LstmCell;
import angland.optimizer.nn.LstmStateTuple;
import angland.optimizer.var.MatrixExpression;
import angland.optimizer.var.ScalarExpression;

public class NGramPredictor {

  private MatrixExpression<String> embedding;
  private LstmCell cell;
  private String[] indexToToken;
  private Map<String, Integer> tokenToIndex;
  private String eosSymbol;
  private String unkSymbol;

  public List<String> predictNext(List<String> input) {
    List<Integer> inputInts =
        input.stream().map(s -> tokenToIndex.getOrDefault(s, tokenToIndex.get(unkSymbol)))
            .collect(Collectors.toList());
    MatrixExpression<String> hiddenState =
        MatrixExpression.repeat(ScalarExpression.constant(0), cell.getSize(), 1);
    MatrixExpression<String> lastOutput =
        MatrixExpression.repeat(ScalarExpression.constant(0), cell.getSize(), 1);
    for (int i : inputInts) {
      MatrixExpression<String> selectedRow = embedding.getRow(ScalarExpression.constant(i));
      LstmStateTuple<String> inputState =
          new LstmStateTuple<String>(hiddenState, selectedRow.transpose());
      hiddenState = cell.apply(inputState).getHiddenState();
    }
    List<String> outputs = new ArrayList<>();
    LstmStateTuple<String> lastState = new LstmStateTuple<>(hiddenState, lastOutput);
    while (outputs.size() == 0 || !outputs.get(outputs.size() - 1).equals(eosSymbol)) {
      LstmStateTuple<String> nextState = cell.apply(lastState);
      nextState.getExposedState();

    }

    return outputs;
  }
}
