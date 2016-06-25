package angland.optimizer.var;

import java.util.HashMap;
import java.util.Map;

/**
 * 
 * @author John Angland
 *
 * @param <VarKey>
 */
public interface Expression<VarKey> {

  /**
   * Evaluates this expression, passing partialSolutions to the evaluateAndCache methods of its
   * descendants.
   * 
   * @param context
   * @param partialSolutions
   * @return
   */
  Calculation<VarKey> evaluate(Map<IndexedKey<VarKey>, Double> context,
      Map<Object, Object> partialSolutions);

  /**
   * Returns the result of this calculation, and stores it in the provided partialSolutions map.
   * 
   * @param context
   * @param partialSolutions
   * @return
   */
  @SuppressWarnings("unchecked")
  default Calculation<VarKey> evaluateAndCache(Map<IndexedKey<VarKey>, Double> context,
      Map<Object, Object> partialSolutions) {
    return (Calculation<VarKey>) partialSolutions.computeIfAbsent(this,
        k -> evaluate(context, partialSolutions));
  }

  public default Calculation<VarKey> evaluate(Map<IndexedKey<VarKey>, Double> ctx) {
    return evaluateAndCache(ctx, new HashMap<>());
  }

  /**
   * Returns an expression with no partial derivatives.
   * 
   * @param value
   * @return
   */
  public static <VarType> Expression<VarType> constant(double value) {
    return (ctx, cache) -> new Calculation<>(value, new HashMap<>(0), ctx);
  }

  /**
   * Returns an expression with a partial derivative of 1 with respect to the key.
   * 
   * @param key
   * @return
   */
  public static <VarType> Expression<VarType> var(VarType key) {
    IndexedKey<VarType> indexedKey = IndexedKey.scalarKey(key);
    return var(indexedKey);
  }

  /**
   * Returns an expression with a partial derivative of 1 with respect to the key.
   * 
   * @param key
   * @return
   */
  public static <VarType> Expression<VarType> var(IndexedKey<VarType> key) {
    return (ctx, cache) -> {
      Map<IndexedKey<VarType>, Double> gradient = new HashMap<>(1, 1);
      gradient.put(key, 1.0);
      if (ctx.get(key) == null) {
        throw new IllegalArgumentException("No context mapping found for variable: " + key);
      }
      return new Calculation<>(ctx.get(key), gradient, ctx);
    };
  }

  /**
   * Add to another expression.
   * 
   * @param other
   * @return
   */
  public default Expression<VarKey> plus(Expression<VarKey> other) {
    return (ctx, cache) -> evaluateAndCache(ctx, cache).plus(other.evaluateAndCache(ctx, cache));
  }

  /**
   * Add to another expression.
   */
  public default Expression<VarKey> minus(Expression<VarKey> other) {
    return plus(other.times(constant(-1)));
  }

  /**
   * Multiply by another expression.
   * 
   * @param other
   * @return
   */
  public default Expression<VarKey> times(Expression<VarKey> other) {
    return (ctx, cache) -> evaluateAndCache(ctx, cache).times(other.evaluateAndCache(ctx, cache));
  }

  /**
   * Divide by another expression
   */
  public default Expression<VarKey> divide(Expression<VarKey> other) {
    return (ctx, cache) -> evaluateAndCache(ctx, cache).times(
        other.power(-1).evaluateAndCache(ctx, cache));
  }


  /**
   * Raise this expression to a power.
   * 
   * @param exponent
   * @return
   */
  public default Expression<VarKey> power(double exponent) {
    return (ctx, cache) -> evaluateAndCache(ctx, cache).power(exponent);
  }

  public default MatrixExpression<VarKey> times(MatrixExpression<VarKey> matrix) {
    return (ctx, cache) -> evaluateAndCache(ctx, cache).times(matrix.evaluateAndCache(ctx, cache));
  }


}
