package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.ContextKey;
import angland.optimizer.var.KeyedDerivative;


public class ArrayCache<VarKey> implements Scalar<VarKey> {

  private final double value;
  private final KeyedDerivative<VarKey>[] derivatives;


  public ArrayCache(double value, KeyedDerivative<VarKey>[] derivatives) {
    super();
    this.value = value;
    this.derivatives = derivatives;
  }

  @Override
  public double value() {
    return value;
  }

  @Override
  public int getBranchComplexity() {
    return 1;
  }

  @Override
  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer) {
    for (int i = 0; i < derivatives.length; ++i) {
      if (derivatives[i] != null && derivatives[i].getValue() != 0) {
        consumer.accept(derivatives[i]);
      }
    }
  }

  @Override
  public double d(ContextKey<VarKey> key) {
    KeyedDerivative<VarKey> kd = derivatives[key.getIdx()];
    if (kd == null) {
      return 0;
    }
    return kd.getValue();
  }


}
