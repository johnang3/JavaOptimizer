package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;

public class ScalarVariable<VarKey> implements Scalar<VarKey> {

  private final IndexedKey<VarKey> key;
  private final double value;
  private final KeyedDerivative<VarKey> keyedDerivative;

  public ScalarVariable(IndexedKey<VarKey> key, double value) {
    super();
    this.key = key;
    this.value = value;
    this.keyedDerivative = new KeyedDerivative<>(key, 1);
    if (Double.isNaN(this.value)) {
      throw new RuntimeException("NaN value");
    }
  }

  @Override
  public double value() {
    return value;
  }

  @Override
  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer) {
    consumer.accept(keyedDerivative);
  }

  @Override
  public double d(IndexedKey<VarKey> key) {
    return key.equals(this.key) ? 1 : 0;
  }

  @Override
  public int getBranchComplexity() {
    return 1;
  }

}
