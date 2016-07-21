package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.ContextKey;
import angland.optimizer.var.KeyedDerivative;

public class ClippedGradientScalar<VarKey> implements IScalarValue<VarKey> {

  private final IScalarValue<VarKey> source;
  private final double clipBelow;
  private final double value;
  private final int branchComplexity;

  public ClippedGradientScalar(IScalarValue<VarKey> source, double clipBelow) {
    super();
    this.source = source;
    this.clipBelow = clipBelow;
    this.branchComplexity = source.getBranchComplexity();
    this.value = source.value();
  }

  @Override
  public double value() {
    return value;
  }

  @Override
  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer) {
    source.actOnKeyedDerivatives(kd -> {
      if (Math.abs(kd.getValue()) >= clipBelow) {
        consumer.accept(kd);
      }
    });
  }

  @Override
  public double d(ContextKey<VarKey> key) {
    double deriv = source.d(key);
    return Math.abs(deriv) >= clipBelow ? deriv : 0;
  }

  @Override
  public int getBranchComplexity() {
    return branchComplexity;
  }

}
