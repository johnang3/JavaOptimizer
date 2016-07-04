package angland.optimizer.var;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import angland.optimizer.vec.MathUtils;

/**
 * @author John Angland
 *
 * @param <VarKey>
 */
public class ScalarValue<VarKey> {

  private final double value;
  private final Map<IndexedKey<VarKey>, Double> gradient;

  public static <VarKey> ScalarValue<VarKey> constant(double value) {
    return new ScalarValue<>(value, new HashMap<>(0));
  }

  public static <VarKey> ScalarValue<VarKey> var(VarKey key, Map<IndexedKey<VarKey>, Double> context) {
    return varIndexed(IndexedKey.scalarKey(key), context);
  }

  public static <VarKey> ScalarValue<VarKey> varIndexed(IndexedKey<VarKey> key,
      Map<IndexedKey<VarKey>, Double> context) {
    Map<IndexedKey<VarKey>, Double> gradient = new HashMap<>(1, 1);
    gradient.put(key, 1.0);
    return new ScalarValue<>(context.get(key), gradient);
  }

  ScalarValue(double value, Map<IndexedKey<VarKey>, Double> gradient) {
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

  public ScalarValue<VarKey> plus(ScalarValue<VarKey> other) {
    double newVal = value + other.value();
    Map<IndexedKey<VarKey>, Double> newGrad = MathUtils.add(gradient, other.gradient);
    return new ScalarValue<>(newVal, newGrad);
  }

  public ScalarValue<VarKey> minus(ScalarValue<VarKey> other) {
    return this.plus(other.times(constant(-1)));
  }

  public ScalarValue<VarKey> divide(ScalarValue<VarKey> other) {
    return times(other.power(-1));
  }

  /**
   * Returns e^x, where x is this calculation.
   * 
   * @return
   */
  public ScalarValue<VarKey> exp() {
    double newVal = Math.exp(value);
    Map<IndexedKey<VarKey>, Double> newGrad = new HashMap<>();
    for (Map.Entry<IndexedKey<VarKey>, Double> e : gradient.entrySet()) {
      newGrad.put(e.getKey(), Math.exp(e.getValue()));
    }
    return new ScalarValue<>(newVal, newGrad);
  }

  @SuppressWarnings("unchecked")
  public static <VarKey> ScalarValue<VarKey> sum(ScalarValue<VarKey>... args) {
    if (args.length == 0) {
      throw new IllegalArgumentException("Cannot take the sum of zero elements.");
    }
    double newVal = 0;
    Map<IndexedKey<VarKey>, Double> newGrad = new HashMap<>();
    Consumer<Map.Entry<IndexedKey<VarKey>, Double>> accumulator =
        entry -> newGrad.merge(entry.getKey(), entry.getValue(), Double::sum);
    for (ScalarValue<VarKey> calc : args) {
      newVal += calc.value();
      calc.getGradient().entrySet().forEach(accumulator);
    }
    return new ScalarValue<>(newVal, newGrad);
  }

  public static class Builder<VarKey> {
    private double value = 0;
    private final Map<IndexedKey<VarKey>, Double> gradient;

    public Builder(int gradientVars) {
      gradient = new HashMap<>(gradientVars);
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

    public void increment(ScalarValue<VarKey> other) {
      this.value += other.value();
      other.getGradient().forEach((k, v) -> gradient.merge(k, v, Double::sum));
    }

    public Map<IndexedKey<VarKey>, Double> getGradient() {
      return gradient;
    }

    public ScalarValue<VarKey> build() {
      return new ScalarValue<>(value, gradient);
    }

  }

  public ScalarValue<VarKey> times(ScalarValue<VarKey> other) {
    double newVal = value * other.value();
    Map<IndexedKey<VarKey>, Double> newGrad =
        new HashMap<>(gradient.size() + other.gradient.size(), 1);
    for (Map.Entry<IndexedKey<VarKey>, Double> entry : this.gradient.entrySet()) {
      if (entry.getValue() == 0) continue;
      newGrad.merge(entry.getKey(), entry.getValue() * other.value(), Double::sum);
    }
    for (Map.Entry<IndexedKey<VarKey>, Double> entry : other.getGradient().entrySet()) {
      if (entry.getValue() == 0) continue;
      newGrad.merge(entry.getKey(), entry.getValue() * this.value(), Double::sum);
    }
    return new ScalarValue<>(newVal, newGrad);
  }

  public ScalarValue<VarKey> power(double exponent) {
    double newVal = Math.pow(value, exponent);
    Map<IndexedKey<VarKey>, Double> newGradient = new HashMap<>(gradient.size(), 1);
    gradient.forEach((k, v) -> newGradient.put(k, exponent * Math.pow(value, exponent - 1)));
    return new ScalarValue<>(newVal, newGradient);
  }

  public ScalarValue<VarKey> sigmoid() {
    double newVal = sigmoidVal(value);
    Map<IndexedKey<VarKey>, Double> newGradient = new HashMap<>(gradient.size(), 1);
    gradient.forEach((k, v) -> newGradient.put(k, sigmoidVal(v) * (1 - sigmoidVal(v))));
    return new ScalarValue<>(newVal, newGradient);
  }

  public ScalarValue<VarKey> tanh() {
    double newVal = Math.tanh(value);
    Map<IndexedKey<VarKey>, Double> newGradient = new HashMap<>(gradient.size(), 1);
    gradient.forEach((k, v) -> newGradient.put(k, 1 - Math.pow(Math.tanh(v), 2)));
    return new ScalarValue<>(newVal, newGradient);
  }

  public ScalarValue<VarKey> ln() {
    double newVal = Math.log(value);
    Map<IndexedKey<VarKey>, Double> newGradient = new HashMap<>(gradient.size(), 1);
    gradient.forEach((k, v) -> newGradient.put(k, v / newVal));
    return new ScalarValue<>(newVal, newGradient);
  }

  private double sigmoidVal(double x) {
    return 1.0 / (1 + Math.exp(-x));
  }

  public ArrayMatrixValue<VarKey> times(IMatrixValue<VarKey> matrix) {
    return ArrayMatrixValue.times(this, matrix);
  }

  public String toString() {
    return "Calculation(" + value + " " + gradient.toString() + ")";
  }



}
