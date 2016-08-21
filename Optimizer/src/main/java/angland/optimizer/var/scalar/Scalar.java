package angland.optimizer.var.scalar;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import angland.optimizer.var.DerivativeMap;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;
import angland.optimizer.var.matrix.Matrix;
import angland.optimizer.vec.MathUtils;

/**
 * Encapsulates the value of a scalar and its partial derivatives with respect to any variables.
 * 
 * @author John Angland
 *
 * @param <VarKey>
 */
public interface Scalar<VarKey> {

  public double value();

  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer);

  public double d(IndexedKey<VarKey> key);

  /**
   * Returns the derivative with respect to a scalar of the specified name.
   * 
   * @param key
   * @return
   */
  public default double d(VarKey key) {
    return d(IndexedKey.scalarKey(key));
  }


  public default Map<IndexedKey<VarKey>, Double> getGradient() {
    Map<IndexedKey<VarKey>, Double> gradient = new HashMap<>();
    actOnKeyedDerivatives(kd -> gradient.merge(kd.getKey(), kd.getValue(), Double::sum));
    return gradient;
  }

  public static <VarKey> Scalar<VarKey> constant(double value) {
    return new ScalarConstant<>(value);
  }

  public static <VarKey> Scalar<VarKey> var(IndexedKey<VarKey> key, double value) {
    return new ScalarVariable<>(key, value);
  }

  public static <VarKey> Scalar<VarKey> var(VarKey key, Map<IndexedKey<VarKey>, Double> context) {
    return var(IndexedKey.scalarKey(key), context);
  }

  public static <VarKey> Scalar<VarKey> var(VarKey key, double value) {
    return var(IndexedKey.scalarKey(key), value);
  }

  public static <VarKey> Scalar<VarKey> varOrConst(VarKey key,
      Map<IndexedKey<VarKey>, Double> context, boolean constant) {
    return varOrConst(IndexedKey.scalarKey(key), context, constant);
  }

  public static <VarKey> Scalar<VarKey> varOrConst(IndexedKey<VarKey> key,
      Map<IndexedKey<VarKey>, Double> context, boolean constant) {
    return constant ? constant(context.get(key)) : var(key, context);
  }

  public static <VarKey> Scalar<VarKey> var(IndexedKey<VarKey> key,
      Map<IndexedKey<VarKey>, Double> context) {
    Double val = context.get(key);
    if (val == null) {
      throw new RuntimeException("No context value for key " + key);
    }
    return new ScalarVariable<>(key, val);
  }

  /**
   * Returns 1 if this Scalar contains cached derivatives.
   * 
   * Otherwise, returns the sum of this node's children's branch complexities.
   * 
   * @return
   */
  int getBranchComplexity();

  public default Scalar<VarKey> ln() {
    return new UnaryScalarOperator<VarKey>(Math.log(value()), 1 / value(), this);
  }

  public default Scalar<VarKey> tanh() {
    return new UnaryScalarOperator<>(Math.tanh(value()), 1 - Math.pow(Math.tanh(value()), 2), this);
  }

  public default Scalar<VarKey> power(double exponent) {
    return new UnaryScalarOperator<>(Math.pow(value(), exponent), exponent
        * Math.pow(value(), exponent - 1), this);

  }

  public default Scalar<VarKey> sigmoid() {
    return new UnaryScalarOperator<>(MathUtils.sigmoidVal(value()), MathUtils.sigmoidVal(value())
        * (1 - MathUtils.sigmoidVal(value())), this);

  }


  public default Scalar<VarKey> plus(Scalar<VarKey> other) {
    return new StreamingSum<>(this, other);
  }


  public default Scalar<VarKey> minus(Scalar<VarKey> other) {
    return this.plus(other.times(constant(-1)));
  }

  public default Scalar<VarKey> times(Scalar<VarKey> other) {
    return new StreamingProduct<>(this, other);
  }

  public default Scalar<VarKey> divide(Scalar<VarKey> other) {
    return new StreamingProduct<>(this, other.power(-1));
  }

  public default Scalar<VarKey> cache() {
    return cache(10);
  }

  public default Scalar<VarKey> cache(int cacheSize) {
    DerivativeMap<VarKey> gradient = new DerivativeMap<>(cacheSize);
    actOnKeyedDerivatives(kd -> gradient.merge(kd.getKey(), kd.getValue()));
    return new MappedDerivativeScalar<>(value(), gradient);
  }

  /**
   * Caches only the n entries with the highest absolute value. Discards the rest.
   * 
   * @param limitedCount
   * @return
   */
  public default Scalar<VarKey> discardBeyond(int n) {
    DerivativeMap<VarKey> gradient = new DerivativeMap<>(n);
    actOnKeyedDerivatives(kd -> gradient.merge(kd.getKey(), kd.getValue()));
    return new MappedDerivativeScalar<>(value(), gradient.getHighestAbs(n));
  }

  public default Scalar<VarKey> exp() {
    return new UnaryScalarOperator<>(Math.exp(value()), Math.exp(value()), this);
  }


  public default Matrix<VarKey> times(Matrix<VarKey> matrix) {
    return matrix.transform(i -> i.times(this));
  }

  public default Scalar<VarKey> clipGradient(double cutoff) {
    return new ClippedGradientScalar<>(this, cutoff);
  }



  public default Scalar<VarKey> toConstant() {
    return constant(value());
  }

  /**
   * @param left
   * @param right
   */
  public static <VarKey> Scalar<VarKey> min(Scalar<VarKey> left, Scalar<VarKey> right) {
    return left.value() <= right.value() ? left : right;
  }



}
