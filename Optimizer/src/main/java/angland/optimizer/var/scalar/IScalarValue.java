package angland.optimizer.var.scalar;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import angland.optimizer.var.Context;
import angland.optimizer.var.ContextKey;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;
import angland.optimizer.var.matrix.IMatrixValue;
import angland.optimizer.vec.MathUtils;

public interface IScalarValue<VarKey> {

  public double value();

  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer);

  public double d(ContextKey<VarKey> key);

  public default Map<ContextKey<VarKey>, Double> getGradient() {
    Map<ContextKey<VarKey>, Double> gradient = new HashMap<>();
    actOnKeyedDerivatives(kd -> gradient.merge(kd.getKey(), kd.getValue(), Double::sum));
    return gradient;
  }

  public static <VarKey> IScalarValue<VarKey> constant(double value) {
    return new ScalarConstant<>(value);
  }

  public static <VarKey> IScalarValue<VarKey> var(VarKey key, Context<VarKey> context) {
    return var(context.getContextTemplate().getContextKey(IndexedKey.scalarKey(key)), context);
  }

  public static <VarKey> IScalarValue<VarKey> varOrConst(VarKey key, Context<VarKey> context,
      boolean constant) {
    return varOrConst(context.getContextTemplate().getContextKey(IndexedKey.scalarKey(key)),
        context, constant);
  }

  public static <VarKey> IScalarValue<VarKey> varOrConst(ContextKey<VarKey> key,
      Context<VarKey> context, boolean constant) {
    return constant ? constant(context.get(key)) : var(key, context);
  }

  public static <VarKey> IScalarValue<VarKey> var(ContextKey<VarKey> key, Context<VarKey> context) {
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

  public default IScalarValue<VarKey> ln() {
    return new UnaryScalarOperator<VarKey>(Math.log(value()), 1 / value(), this);
  }

  public default IScalarValue<VarKey> tanh() {
    return new UnaryScalarOperator<>(Math.tanh(value()), 1 - Math.pow(Math.tanh(value()), 2), this);
  }

  public default IScalarValue<VarKey> power(double exponent) {
    return new UnaryScalarOperator<>(Math.pow(value(), exponent), exponent
        * Math.pow(value(), exponent - 1), this);

  }

  public default IScalarValue<VarKey> sigmoid() {
    return new UnaryScalarOperator<>(MathUtils.sigmoidVal(value()), MathUtils.sigmoidVal(value())
        * (1 - MathUtils.sigmoidVal(value())), this);

  }


  public default IScalarValue<VarKey> plus(IScalarValue<VarKey> other) {
    return new StreamingSum<>(this, other);
  }


  public default IScalarValue<VarKey> minus(IScalarValue<VarKey> other) {
    return this.plus(other.times(constant(-1)));
  }

  public default IScalarValue<VarKey> times(IScalarValue<VarKey> other) {
    return new StreamingProduct<>(this, other);
  }

  public default IScalarValue<VarKey> divide(IScalarValue<VarKey> other) {
    return new StreamingProduct<>(this, other.power(-1));
  }

  public default IScalarValue<VarKey> cache() {
    return new MappedDerivativeScalar<>(value(), getGradient());
  }

  public default IScalarValue<VarKey> exp() {
    return new UnaryScalarOperator<>(Math.exp(value()), Math.exp(value()), this);
  }


  public default IMatrixValue<VarKey> times(IMatrixValue<VarKey> matrix) {
    return matrix.transform(i -> i.times(this));
  }

  public default IScalarValue<VarKey> clipGradient(double cutoff) {
    return new ClippedGradientScalar<>(this, cutoff);
  }

  public default IScalarValue<VarKey> arrayCache(Context<VarKey> context) {
    float[] derivs = getDerivatives(context.getContextTemplate().size());
    Map<ContextKey<VarKey>, Double> gradient = new HashMap<>();
    for (int i = 0; i < derivs.length; ++i) {
      if (derivs[i] != 0) {
        gradient.put(context.getContextTemplate().getKey(i), (double) derivs[i]);
      }
    }
    return new MappedDerivativeScalar<>(value(), gradient);
  }

  public default float[] getDerivatives(int contextSize) {
    float[] arr = new float[contextSize];
    this.actOnKeyedDerivatives(kd -> {
      arr[kd.getKey().getIdx()] += kd.getValue();
    });
    return arr;
  }



}
