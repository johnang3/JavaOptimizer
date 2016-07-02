package angland.optimizer.nn;

import angland.optimizer.var.MatrixExpression;

public class LstmStateTuple<VarKey> {

  private final MatrixExpression<VarKey> hiddenState;
  private final MatrixExpression<VarKey> exposedState;

  public LstmStateTuple(MatrixExpression<VarKey> hiddenState, MatrixExpression<VarKey> exposedState) {
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



}
