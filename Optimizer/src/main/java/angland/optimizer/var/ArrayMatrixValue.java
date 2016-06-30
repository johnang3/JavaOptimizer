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
