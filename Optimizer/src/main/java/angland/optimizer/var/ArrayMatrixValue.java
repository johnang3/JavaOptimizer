package angland.optimizer.var;

import angland.optimizer.var.scalar.IScalarValue;



public class ArrayMatrixValue<VarKey> extends MatrixBase<IScalarValue<VarKey>>
    implements
      IMatrixValue<VarKey> {

  private final IScalarValue<VarKey>[] values;


  protected ArrayMatrixValue(int height, int width, IScalarValue<VarKey>[] values) {
    super(height, width);
    this.values = values;
  }

  @Override
  protected IScalarValue<VarKey>[] values() {
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
        newMatrix.set(i, j, this.get(i, j).plus(other.get(i, j)));
      }
    }
    return newMatrix.build();
  }

  public static <VarKey> ArrayMatrixValue<VarKey> times(IScalarValue<VarKey> scalar,
      IMatrixValue<VarKey> matrix) {
    Builder<VarKey> newMatrix = new Builder<>(matrix.getHeight(), matrix.getWidth());
    for (int i = 0; i < matrix.getHeight(); ++i) {
      for (int j = 0; j < matrix.getWidth(); ++j) {
        newMatrix.set(i, j, matrix.get(i, j).times(scalar));
      }
    }
    return newMatrix.build();
  }

  public static class Builder<VarKey> extends MatrixBase<IScalarValue<VarKey>> {

    protected final IScalarValue<VarKey>[] values;

    @SuppressWarnings("unchecked")
    public Builder(int height, int width) {
      super(height, width);
      this.values = (IScalarValue<VarKey>[]) new IScalarValue[height * width];
    }

    @Override
    protected IScalarValue<VarKey>[] values() {
      return values;
    }

    public void set(int row, int column, IScalarValue<VarKey> calc) {
      validateCoords(row, column);
      values()[column + getWidth() * row] = calc;
    }

    public ArrayMatrixValue<VarKey> build() {
      return new ArrayMatrixValue<>(getHeight(), getWidth(), values);
    }

  }



}
