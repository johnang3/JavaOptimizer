package angland.optimizer.var;

import angland.optimizer.var.scalar.IScalarValue;

public class MatrixRangeView<VarKey> implements IMatrixValue<VarKey> {

  private final IMatrixValue<VarKey> source;
  private final int startRow;
  private final int startCol;
  private final int height;
  private final int width;

  public MatrixRangeView(IMatrixValue<VarKey> source, int startRow, int startCol, int height,
      int width) {
    super();
    if (startRow > source.getHeight()) {
      throw new IllegalArgumentException("Target start row " + startRow
          + " greater than source height " + source.getHeight());
    }
    if (startCol > source.getWidth()) {
      throw new IllegalArgumentException("Target start col " + startCol
          + " greater than source width " + source.getWidth());
    }
    this.source = source;
    this.startRow = startRow;
    this.startCol = startCol;
    this.height = height;
    this.width = width;
  }

  @Override
  public int getHeight() {
    return height;
  }

  @Override
  public int getWidth() {
    return width;
  }

  @Override
  public IScalarValue<VarKey> get(int row, int column) {
    return source.get(startRow + row, startCol + column);
  }



}
