package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.ContextKey;
import angland.optimizer.var.DerivativeMap;
import angland.optimizer.var.KeyedDerivative;

/**
 * @author John Angland
 *
 * @param <VarKey>
 */
public class MappedDerivativeScalar<VarKey> implements IScalarValue<VarKey> {

  private final double value;
  private final DerivativeMap<VarKey> gradient;

  MappedDerivativeScalar(double value, DerivativeMap<VarKey> gradient) {
    super();
    this.value = value;
    this.gradient = gradient;
    if (Double.isNaN(this.value)) {
      throw new RuntimeException("NaN value");
    }
    gradient.actOnEntries(entry -> {
      if (Double.isNaN(entry.getValue())) {
        throw new RuntimeException("NaN derivative.");
      }
    });
  }

  public double value() {
    return value;
  }

  @Override
  public double d(ContextKey<VarKey> v) {
    return gradient.get(v);
  }


  @Override
  public MappedDerivativeScalar<VarKey> cache() {
    return this;
  }



  public static class Builder<VarKey> {
    private double value = 0;
    private final DerivativeMap<VarKey> gradient;

    public Builder(int gradientVars) {
      gradient = new DerivativeMap<VarKey>(10);
    }

    public double getValue() {
      return value;
    }

    public void setValue(double value) {
      this.value = value;
    }

    public void incrementValue(double value) {
      this.value += value;
    }

    public void increment(IScalarValue<VarKey> other) {
      this.value += other.value();
      other.actOnKeyedDerivatives((kd) -> gradient.merge(kd.getKey(), kd.getValue()));
    }

    public DerivativeMap<VarKey> getGradient() {
      return gradient;
    }

    public MappedDerivativeScalar<VarKey> build() {
      return new MappedDerivativeScalar<>(value, gradient);
    }

  }


  public String toString() {
    return "Calculation(" + value + " " + gradient.toString() + ")";
  }

  @Override
  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer) {
    gradient.actOnEntries(consumer);
  }

  @Override
  public int getBranchComplexity() {
    return 1;
  }



}
