package angland.optimizer.var;

import java.util.Map;


public class MatrixValue<VarKey> extends MatrixBase<VarKey> {

  private final Calculation<VarKey>[] values;
  private final Map<IndexedKey<VarKey>, Double> context;

  public Map<IndexedKey<VarKey>, Double> getContext() {
    return context;
  }

  private MatrixValue(int height, int width, Calculation<VarKey>[] values,
      Map<IndexedKey<VarKey>, Double> context) {
    super(height, width);
    this.values = values;
    this.context = context;
  }

  @Override
  protected Calculation<VarKey>[] values() {
    return values;
  }

  public MatrixValue<VarKey> add(MatrixValue<VarKey> other) {
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

  public static <VarKey> MatrixValue<VarKey> times(Calculation<VarKey> scalar,
      MatrixValue<VarKey> matrix) {
    Builder<VarKey> newMatrix = new Builder<>(matrix.getHeight(), matrix.getWidth());
    for (int i = 0; i < matrix.getHeight(); ++i) {
      for (int j = 0; j < matrix.getHeight(); ++j) {
        newMatrix.set(i, j, matrix.getCalculation(i, j).times(scalar));
      }
    }
    return newMatrix.build(scalar.getContext());
  }

  public MatrixValue<VarKey> times(MatrixValue<VarKey> other) {
    if (this.getWidth() != other.getHeight()) {
      throw new IllegalArgumentException("Width of left matrix must equal height of right matrix.");
    }
    Builder<VarKey> builder = new Builder<>(this.getHeight(), other.getWidth());
    for (int i = 0; i < this.getHeight(); ++i) {
      for (int j = 0; j < other.getWidth(); ++j) {
        Calculation.Builder<VarKey> sumBuilder = new Calculation.Builder<>();
        for (int k = 0; k < this.getWidth(); ++k) {
          Calculation<VarKey> x = this.getCalculation(i, k).times(other.getCalculation(k, j));
          sumBuilder.increment(x);
        }
        builder.set(i, j, sumBuilder.build(context));
      }
    }
    return builder.build(context);
  }

  public static class Builder<VarKey> extends MatrixBase<VarKey> {

    private final Calculation<VarKey>[] values;

    @SuppressWarnings("unchecked")
    public Builder(int height, int width) {
      super(height, width);
      this.values = (Calculation<VarKey>[]) new Calculation[height * width];
    }

    @Override
    protected Calculation<VarKey>[] values() {
      return values;
    }

    public void set(int row, int column, Calculation<VarKey> calc) {
      validateCoords(row, column);
      values()[column + getWidth() * row] = calc;
    }

    public MatrixValue<VarKey> build(Map<IndexedKey<VarKey>, Double> context) {
      return new MatrixValue<>(getHeight(), getWidth(), values, context);
    }

  }



}
