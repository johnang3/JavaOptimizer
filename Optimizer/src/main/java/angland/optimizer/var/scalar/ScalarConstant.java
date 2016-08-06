package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.ContextKey;
import angland.optimizer.var.KeyedDerivative;

public class ScalarConstant<VarKey> implements IScalarValue<VarKey> {

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
  public double d(ContextKey<VarKey> key) {
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
