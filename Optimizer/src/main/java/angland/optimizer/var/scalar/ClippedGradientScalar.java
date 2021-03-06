package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;

public class ClippedGradientScalar<VarKey> implements Scalar<VarKey> {

  private final Scalar<VarKey> source;
  private final double clipBelow;
  private final double value;
  private final int branchComplexity;

  public ClippedGradientScalar(Scalar<VarKey> source, double clipBelow) {
    super();
    this.source = source;
    this.clipBelow = clipBelow;
    this.branchComplexity = source.getBranchComplexity();
    this.value = source.value();
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
    source.actOnKeyedDerivatives(kd -> {
      if (Math.abs(kd.getValue()) >= clipBelow) {
        consumer.accept(kd);
      }
    });
  }

  @Override
  public double d(IndexedKey<VarKey> key) {
    double deriv = source.d(key);
    return Math.abs(deriv) >= clipBelow ? deriv : 0;
  }

  @Override
  public int getBranchComplexity() {
    return branchComplexity;
  }

}
