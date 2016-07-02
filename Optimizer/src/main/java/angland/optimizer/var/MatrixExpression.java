package angland.optimizer.var;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public interface MatrixExpression<VarKey> {

  public IMatrixValue<VarKey> evaluate(Map<IndexedKey<VarKey>, Double> context,
      Map<Object, Object> cache);

  @SuppressWarnings("unchecked")
  public default IMatrixValue<VarKey> evaluateAndCache(Map<IndexedKey<VarKey>, Double> context,
      Map<Object, Object> partialSolutions) {
    return (IMatrixValue<VarKey>) partialSolutions.computeIfAbsent(this,
        k -> evaluate(context, partialSolutions));
  }


  public default IMatrixValue<VarKey> evaluate(Map<IndexedKey<VarKey>, Double> ctx) {
    return evaluateAndCache(ctx, new HashMap<>());
  }


  public static <VarKey> MatrixExpression<VarKey> constant(ArrayMatrixValue.Builder<VarKey> builder) {
    return (ctx, cache) -> builder.build();
  }

  public static <VarKey> MatrixExpression<VarKey> variable(VarKey varKey, int height, int width) {
    return (ctx, cache) -> {
      ArrayMatrixValue.Builder<VarKey> builder = new ArrayMatrixValue.Builder<>(height, width);
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          builder.set(i, j,
              ScalarExpression.var(IndexedKey.matrixKey(varKey, i, j)).evaluate(ctx, cache));
        }
      }
      return builder.build();
    };
  }

  public static <VarKey> MatrixExpression<VarKey> repeat(ScalarExpression<VarKey> exp, int height,
      int width) {
    return (ctx, cache) -> {
      ArrayMatrixValue.Builder<VarKey> builder = new ArrayMatrixValue.Builder<>(height, width);
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          builder.set(i, j, exp.evaluateAndCache(ctx, cache));
        }
      }
      return builder.build();
    };
  }

  public default MatrixExpression<VarKey> getRow(ScalarExpression<VarKey> rowIdx) {
    return (ctx, cache) -> {
      int row = (int) rowIdx.evaluateAndCache(ctx, cache).value();
      IMatrixValue<VarKey> solved = this.evaluateAndCache(ctx, cache);
      return new MatrixRangeView<>(solved, row, 0, 1, solved.getWidth());
    };
  }

  public default MatrixExpression<VarKey> plus(MatrixExpression<VarKey> other) {
    return (ctx, cache) -> this.evaluateAndCache(ctx, cache)
        .add(other.evaluateAndCache(ctx, cache));
  }

  public default MatrixExpression<VarKey> pointwiseMultiply(MatrixExpression<VarKey> other) {
    return (ctx, cache) -> this.evaluateAndCache(ctx, cache).multiplyPointwise(
        other.evaluateAndCache(ctx, cache));
  }

  public default MatrixExpression<VarKey> times(MatrixExpression<VarKey> other) {
    return (ctx, cache) -> this.evaluateAndCache(ctx, cache).times(
        other.evaluateAndCache(ctx, cache));
  }

  public default MatrixExpression<VarKey> transpose() {
    return (ctx, cache) -> this.evaluateAndCache(ctx, cache).transpose();
  }

  public default MatrixExpression<VarKey> addToAll(ScalarExpression<VarKey> scalar) {
    return (ctx, cache) -> {
      IMatrixValue<VarKey> val = this.evaluateAndCache(ctx, cache);
      ScalarValue<VarKey> added = scalar.evaluateAndCache(ctx, cache);
      ArrayMatrixValue.Builder<VarKey> builder =
          new ArrayMatrixValue.Builder<>(val.getHeight(), val.getWidth());
      for (int i = 0; i < builder.getHeight(); ++i) {
        for (int j = 0; j < builder.getWidth(); ++j) {
          builder.set(i, j, val.get(i, j).plus(added));
        }
      }
      return builder.build();
    };
  }

  public default MatrixExpression<VarKey> transform(
      Function<ScalarValue<VarKey>, ScalarValue<VarKey>> transformation) {
    return (ctx, cache) -> {
      IMatrixValue<VarKey> val = this.evaluateAndCache(ctx, cache);
      ArrayMatrixValue.Builder<VarKey> builder =
          new ArrayMatrixValue.Builder<>(val.getHeight(), val.getWidth());
      for (int i = 0; i < builder.getHeight(); ++i) {
        for (int j = 0; j < builder.getWidth(); ++j) {
          builder.set(i, j, transformation.apply(val.get(i, j)));
        }
      }
      return builder.build();
    };
  }

}
