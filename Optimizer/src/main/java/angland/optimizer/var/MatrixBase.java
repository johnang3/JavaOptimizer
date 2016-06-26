package angland.optimizer.var;



abstract class MatrixBase<VarType> {

  private final int height;
  private final int width;

  public MatrixBase(int height, int width) {
    super();
    this.height = height;
    this.width = width;
  }

  /**
   * 
   * @return
   */
  protected abstract ScalarValue<VarType>[] values();

  public int getHeight() {
    return height;
  }

  public int getWidth() {
    return width;
  }

  public ScalarValue<VarType> getCalculation(int row, int column) {
    validateCoords(row, column);
    return values()[column + getWidth() * row];
  }

  public void validateCoords(int row, int column) {
    if (row < 0) {
      throw new IllegalArgumentException("row may not be less than zero.");
    }
    if (column < 0) {
      throw new IllegalArgumentException("column may not be less than zero");
    }
    if (row >= height) {
      throw new IllegalArgumentException("row (" + row + ") must be less than height (" + height
          + ")");
    }
    if (column >= width) {
      throw new IllegalArgumentException("column (" + column + ") must be less than width ("
          + width + ")");
    }
  }


}
