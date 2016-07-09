package angland.optimizer.var.scalar;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;

/**
 * @author John Angland
 *
 * @param <VarKey>
 */
public class MappedDerivativeScalar<VarKey> implements IScalarValue<VarKey> {

  private final double value;
  private final Map<IndexedKey<VarKey>, Double> gradient;

  MappedDerivativeScalar(double value, Map<IndexedKey<VarKey>, Double> gradient) {
    super();
    this.value = value;
    this.gradient = Collections.unmodifiableMap(gradient);
  }

  public double value() {
    return value;
  }

  public double d(IndexedKey<VarKey> v) {
    return gradient.getOrDefault(v, 0.0);
  }

  public Map<IndexedKey<VarKey>, Double> getGradient() {
    return gradient;
  }

  @Override
  public MappedDerivativeScalar<VarKey> cache() {
    return this;
  }



  @SuppressWarnings("unchecked")
  public static <VarKey> MappedDerivativeScalar<VarKey> sum(MappedDerivativeScalar<VarKey>... args) {
    if (args.length == 0) {
      throw new IllegalArgumentException("Cannot take the sum of zero elements.");
    }
    double newVal = 0;
    Map<IndexedKey<VarKey>, Double> newGrad = new HashMap<>();
    Consumer<Map.Entry<IndexedKey<VarKey>, Double>> accumulator =
        entry -> newGrad.merge(entry.getKey(), entry.getValue(), Double::sum);
    for (MappedDerivativeScalar<VarKey> calc : args) {
      newVal += calc.value();
      calc.getGradient().entrySet().forEach(accumulator);
    }
    return new MappedDerivativeScalar<>(newVal, newGrad);
  }

  public static class Builder<VarKey> {
    private double value = 0;
    private final Map<IndexedKey<VarKey>, Double> gradient;

    public Builder(int gradientVars) {
      gradient = new HashMap<>(gradientVars, 1);
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
      other.actOnKeyedDerivatives((kd) -> gradient.merge(kd.getKey(), kd.getValue(), Double::sum));
    }

    public Map<IndexedKey<VarKey>, Double> getGradient() {
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
    for (Map.Entry<IndexedKey<VarKey>, Double> e : this.gradient.entrySet()) {
      consumer.accept(new KeyedDerivative<>(e.getKey(), e.getValue()));
    }
  }

  @Override
  public int getBranchComplexity() {
    return 1;
  }



}
