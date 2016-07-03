package angland.optimizer.nn;

import java.util.HashMap;
import java.util.Map;

import angland.optimizer.var.IMatrixValue;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.MatrixExpression;

public class LstmStateTupleExpression<VarKey> {

  private final MatrixExpression<VarKey> hiddenState;
  private final MatrixExpression<VarKey> exposedState;

  public LstmStateTupleExpression(MatrixExpression<VarKey> hiddenState,
      MatrixExpression<VarKey> exposedState) {
    super();
    this.hiddenState = hiddenState;
    this.exposedState = exposedState;
  }

  public MatrixExpression<VarKey> getHiddenState() {
    return hiddenState;
  }

  public MatrixExpression<VarKey> getExposedState() {
    return exposedState;
  }

  @SuppressWarnings("unchecked")
  public LstmStateTupleValue<VarKey> evaluate(Map<IndexedKey<VarKey>, Double> context) {
    Map<Object, Object> partialSolutions = new HashMap<>();
    IMatrixValue<VarKey> exposedSolution = exposedState.evaluate(context, partialSolutions);
    IMatrixValue<VarKey> hiddenSolution = (IMatrixValue<VarKey>) partialSolutions.get(hiddenState);
    return new LstmStateTupleValue<>(hiddenSolution, exposedSolution);
  }
}
