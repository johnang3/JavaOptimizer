package angland.optimizer.var;

import java.util.Map;


public class ArrayMatrixValue<VarKey> extends MatrixBase<VarKey> implements IMatrixValue<VarKey> {

  private final ScalarValue<VarKey>[] values;
  private final Map<IndexedKey<VarKey>, Double> context;

  public Map<IndexedKey<VarKey>, Double> getContext() {
    return context;
  }

  protected ArrayMatrixValue(int height, int width, ScalarValue<VarKey>[] values,
      Map<IndexedKey<VarKey>, Double> context) {
    super(height, width);
    this.values = values;
    this.context = context;
  }

  @Override
  protected ScalarValue<VarKey>[] values() {
    return values;
  }

  public ArrayMatrixValue<VarKey> add(ArrayMatrixValue<VarKey> other) {
    if (this.getHeight() != other.getHeight()) {
      throw new RuntimeException("Cannot add matrices of differing heights.");
    }
    if (this.getWidth() != other.getWidth()) {
      throw new RuntimeException("Cannot add matrices of differing widths");
    }
    Builder<VarKey> newMatrix = new Builder<>(getHeight(), getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getHeight(); ++j) {
        newMatrix.set(i, j, this.getCalculation(i, j).plus(other.getCalculation(i, j)));
      }
    }
    return newMatrix.build(other.getContext());
  }

  public static <VarKey> ArrayMatrixValue<VarKey> times(ScalarValue<VarKey> scalar,
      IMatrixValue<VarKey> matrix) {
    Builder<VarKey> newMatrix = new Builder<>(matrix.getHeight(), matrix.getWidth());
    for (int i = 0; i < matrix.getHeight(); ++i) {
      for (int j = 0; j < matrix.getHeight(); ++j) {
        newMatrix.set(i, j, matrix.getCalculation(i, j).times(scalar));
      }
    }
    return newMatrix.build(scalar.getContext());
  }

  public ArrayMatrixValue<VarKey> times(ArrayMatrixValue<VarKey> other) {
    if (this.getWidth() != other.getHeight()) {
      throw new IllegalArgumentException("Width of left matrix (" + getWidth()
          + ") must equal height of right matrix (" + other.getHeight() + ")");
    }
    Builder<VarKey> builder = new Builder<>(this.getHeight(), other.getWidth());
    for (int i = 0; i < this.getHeight(); ++i) {
      for (int j = 0; j < other.getWidth(); ++j) {
        ScalarValue.Builder<VarKey> sumBuilder = new ScalarValue.Builder<>(this.getWidth() * 3);
        for (int k = 0; k < this.getWidth(); ++k) {
          ScalarValue<VarKey> left = this.getCalculation(i, k);
          ScalarValue<VarKey> right = other.getCalculation(k, j);
          sumBuilder.incrementValue(left.value() * right.value());
          for (Map.Entry<IndexedKey<VarKey>, Double> entry : left.getGradient().entrySet()) {
            if (entry.getValue() != 0) {
              sumBuilder.getGradient().merge(entry.getKey(), entry.getValue() * right.value(),
                  Double::sum);
            }
          }
          for (Map.Entry<IndexedKey<VarKey>, Double> entry : right.getGradient().entrySet()) {
            if (entry.getValue() != 0) {
              sumBuilder.getGradient().merge(entry.getKey(), entry.getValue() * left.value(),
                  Double::sum);
            }
          }
        }
        builder.set(i, j, sumBuilder.build(context));
      }
    }
    return builder.build(context);
  }

  public static class Builder<VarKey> extends MatrixBase<VarKey> {

    protected final ScalarValue<VarKey>[] values;

    @SuppressWarnings("unchecked")
    public Builder(int height, int width) {
      super(height, width);
      this.values = (ScalarValue<VarKey>[]) new ScalarValue[height * width];
    }

    @Override
    protected ScalarValue<VarKey>[] values() {
      return values;
    }

    public void set(int row, int column, ScalarValue<VarKey> calc) {
      validateCoords(row, column);
      values()[column + getWidth() * row] = calc;
    }

    public ArrayMatrixValue<VarKey> build(Map<IndexedKey<VarKey>, Double> context) {
      return new ArrayMatrixValue<>(getHeight(), getWidth(), values, context);
    }

  }



}
