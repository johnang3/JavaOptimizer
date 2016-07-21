package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.ContextKey;
import angland.optimizer.var.KeyedDerivative;

public class ScalarVariable<VarKey> implements IScalarValue<VarKey> {

  private final ContextKey<VarKey> key;
  private final double value;
  private final KeyedDerivative<VarKey> keyedDerivative;

  public ScalarVariable(ContextKey<VarKey> key, double value) {
    super();
    this.key = key;
    this.value = value;
    this.keyedDerivative = new KeyedDerivative<>(key, 1);
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
  public double d(ContextKey<VarKey> key) {
    return key.equals(this.key) ? 1 : 0;
  }

  @Override
  public int getBranchComplexity() {
    return 1;
  }

}
