package angland.optimizer.nn;

import angland.optimizer.var.IMatrixValue;

public class LstmStateTupleValue<VarKey> {

  private final IMatrixValue<VarKey> hiddenValue;
  private final IMatrixValue<VarKey> exposedState;

  public LstmStateTupleValue(IMatrixValue<VarKey> hiddenValue, IMatrixValue<VarKey> exposedState) {
    super();
    this.hiddenValue = hiddenValue;
    this.exposedState = exposedState;
  }

  public IMatrixValue<VarKey> getHiddenState() {
    return hiddenValue;
  }

  public IMatrixValue<VarKey> getExposedState() {
    return exposedState;
  }

  public LstmStateTupleExpression<VarKey> toConstant() {
    return new LstmStateTupleExpression<>(hiddenValue.toConstant(), exposedState.toConstant());
  }


}
