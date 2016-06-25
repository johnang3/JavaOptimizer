package angland.optimizer.var;

import java.util.HashMap;
import java.util.Map;

public interface MatrixExpression<VarKey> {

  public MatrixValue<VarKey> evaluate(Map<IndexedKey<VarKey>, Double> context,
      Map<Object, Object> cache);

  @SuppressWarnings("unchecked")
  public default MatrixValue<VarKey> evaluateAndCache(Map<IndexedKey<VarKey>, Double> context,
      Map<Object, Object> partialSolutions) {
    return (MatrixValue<VarKey>) partialSolutions.computeIfAbsent(this,
        k -> evaluate(context, partialSolutions));
  }


  public default MatrixValue<VarKey> evaluate(Map<IndexedKey<VarKey>, Double> ctx) {
    return evaluateAndCache(ctx, new HashMap<>());
  }


  public static <VarKey> MatrixExpression<VarKey> constant(MatrixValue.Builder<VarKey> builder) {
    return (ctx, cache) -> builder.build(ctx);
  }

  public static <VarKey> MatrixExpression<VarKey> variable(VarKey varKey, int height, int width) {
    return (ctx, cache) -> {
      MatrixValue.Builder<VarKey> builder = new MatrixValue.Builder<>(height, width);
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          builder
              .set(i, j, Expression.var(IndexedKey.matrixKey(varKey, i, j)).evaluate(ctx, cache));
        }
      }
      return builder.build(ctx);
    };
  }


  public default MatrixExpression<VarKey> plus(MatrixExpression<VarKey> other) {
    return (ctx, cache) -> this.evaluateAndCache(ctx, cache)
        .add(other.evaluateAndCache(ctx, cache));
  }

  public default MatrixExpression<VarKey> times(MatrixExpression<VarKey> other) {
    return (ctx, cache) -> this.evaluateAndCache(ctx, cache).times(
        other.evaluateAndCache(ctx, cache));
  }


}
