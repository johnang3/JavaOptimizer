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

  public default MatrixExpression<VarKey> getColumn(ScalarExpression<VarKey> colIdx) {
    return (ctx, cache) -> {
      int col = (int) colIdx.evaluateAndCache(ctx, cache).value();
      IMatrixValue<VarKey> solved = this.evaluateAndCache(ctx, cache);
      return new MatrixRangeView<>(solved, 0, col, solved.getHeight(), 1);
    };
  }

  /**
   * Returns a new matrix m of dimensions (0, this.getWidth()).
   * 
   * Each entry m[0][i] is equal to the l2 norm of the i'th column of this matrix minus target.
   * 
   * @param target
   * @return
   */
  public default MatrixExpression<VarKey> columnProximity(MatrixExpression<VarKey> target) {
    return (ctx, cache) -> {
      IMatrixValue<VarKey> solved = this.evaluateAndCache(ctx, cache);
      ArrayMatrixValue.Builder<VarKey> builder =
          new ArrayMatrixValue.Builder<>(1, solved.getWidth());
      for (int i = 0; i < solved.getWidth(); ++i) {
        ScalarValue.Builder<VarKey> columnBuilder = new ScalarValue.Builder<>(1);
        for (int j = 0; j < solved.getHeight(); ++j) {
          columnBuilder.increment(solved.get(j, i).power(2));
        }
        builder.set(0, i, columnBuilder.build().power(.5));
      }
      return builder.build();
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

  public default MatrixExpression<VarKey> softmax() {
    return (ctx, cache) -> {
      return this.evaluateAndCache(ctx, cache).softmax();
    };
  }

  public default ScalarExpression<VarKey> maxIdx() {
    return (ctx, cache) -> {
      return this.evaluateAndCache(ctx, cache).maxIdx();
    };
  }

}
