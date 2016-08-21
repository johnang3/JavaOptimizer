package angland.optimizer.nn;

import angland.optimizer.var.matrix.Matrix;

public class RnnStateTuple<VarKey> {

  private final Matrix<VarKey> hiddenValue;
  private final Matrix<VarKey> exposedState;

  public RnnStateTuple(Matrix<VarKey> hiddenValue, Matrix<VarKey> exposedState) {
    super();
    this.hiddenValue = hiddenValue;
    this.exposedState = exposedState;
  }

  public Matrix<VarKey> getHiddenState() {
    return hiddenValue;
  }

  public Matrix<VarKey> getExposedState() {
    return exposedState;
  }

  public RnnStateTuple<VarKey> toConstant() {
    return new RnnStateTuple<>(hiddenValue.toConstant(), exposedState.toConstant());
  }


}
