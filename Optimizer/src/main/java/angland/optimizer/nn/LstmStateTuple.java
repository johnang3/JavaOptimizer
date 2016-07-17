package angland.optimizer.nn;

import angland.optimizer.var.matrix.IMatrixValue;

public class LstmStateTuple<VarKey> {

  private final IMatrixValue<VarKey> hiddenValue;
  private final IMatrixValue<VarKey> exposedState;

  public LstmStateTuple(IMatrixValue<VarKey> hiddenValue, IMatrixValue<VarKey> exposedState) {
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

  public LstmStateTuple<VarKey> toConstant() {
    return new LstmStateTuple<>(hiddenValue.toConstant(), exposedState.toConstant());
  }


}
