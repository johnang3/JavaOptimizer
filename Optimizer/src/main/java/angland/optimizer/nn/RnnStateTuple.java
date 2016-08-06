package angland.optimizer.nn;

import angland.optimizer.var.matrix.IMatrixValue;

public class RnnStateTuple<VarKey> {

  private final IMatrixValue<VarKey> hiddenValue;
  private final IMatrixValue<VarKey> exposedState;

  public RnnStateTuple(IMatrixValue<VarKey> hiddenValue, IMatrixValue<VarKey> exposedState) {
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

  public RnnStateTuple<VarKey> toConstant() {
    return new RnnStateTuple<>(hiddenValue.toConstant(), exposedState.toConstant());
  }


}
