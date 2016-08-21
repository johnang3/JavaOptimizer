package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;

class ScalarConstant<VarKey> implements Scalar<VarKey> {

  private final double value;

  public ScalarConstant(double value) {
    this.value = value;
    if (Double.isNaN(this.value)) {
      throw new RuntimeException("NaN value");
    }
  }

  @Override
  public double value() {
    return value;
  }

  @Override
  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer) {}

  @Override
  public double d(IndexedKey<VarKey> key) {
    return 0;
  }

  @Override
  public int getBranchComplexity() {
    return 0;
  }

  public String toString() {
    return Double.toString(value);
  }

}
