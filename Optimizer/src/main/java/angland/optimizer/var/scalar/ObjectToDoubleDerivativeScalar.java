package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.utils.ObjectToDoubleMap;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;

public class ObjectToDoubleDerivativeScalar<VarKey> implements IScalarValue<VarKey> {

  private final double value;
  private final ObjectToDoubleMap<IndexedKey<VarKey>> derivs;

  public ObjectToDoubleDerivativeScalar(IScalarValue<VarKey> parent) {
    super();
    this.value = parent.value();
    this.derivs = new ObjectToDoubleMap<>(10);
    parent.actOnKeyedDerivatives(deriv -> derivs.adjust(deriv.getKey(), deriv.getValue()));
  }

  @Override
  public double value() {
    return value;
  }

  @Override
  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer) {
    derivs.actOnEntries(entry -> {
      consumer.accept(new KeyedDerivative<>(entry.getKey(), entry.getValue()));
    });
  }

  @Override
  public double d(IndexedKey<VarKey> key) {
    return derivs.get(key);
  }

  @Override
  public int getBranchComplexity() {
    return 1;
  }

}
