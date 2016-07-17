package angland.optimizer.var.matrix;

import angland.optimizer.var.scalar.IScalarValue;


public class TransposeView<VarKey> implements IMatrixValue<VarKey> {

  private final IMatrixValue<VarKey> source;

  public TransposeView(IMatrixValue<VarKey> source) {
    super();
    this.source = source;
  }

  @Override
  public int getHeight() {
    return source.getWidth();
  }

  @Override
  public int getWidth() {
    return source.getHeight();
  }

  @Override
  public IScalarValue<VarKey> get(int row, int column) {
    return source.get(column, row);
  }


  @Override
  public IMatrixValue<VarKey> transpose() {
    return source;
  }

}
