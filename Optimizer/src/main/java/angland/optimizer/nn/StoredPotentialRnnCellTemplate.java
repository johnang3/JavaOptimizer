package angland.optimizer.nn;

import java.util.stream.Stream;

import angland.optimizer.var.Context;
import angland.optimizer.var.IndexedKey;

public class StoredPotentialRnnCellTemplate implements RnnCellTemplate {

  private final String varPrefix;
  private final int size;
  private final double gradientClipThreshold;
  private final boolean isConstant;


  public StoredPotentialRnnCellTemplate(String varPrefix, int size, double gradientClipThreshold,
      boolean isConstant) {
    super();
    this.varPrefix = varPrefix;
    this.size = size;
    this.gradientClipThreshold = gradientClipThreshold;
    this.isConstant = isConstant;
  }

  @Override
  public RnnCell<String> create(Context<String> context) {
    return new StoredPotentialRnnCell(varPrefix, size, context, gradientClipThreshold, isConstant);
  }

  @Override
  public Stream<IndexedKey<String>> getKeys() {
    return Stream.concat(IndexedKey.getAllMatrixKeys("adjust_weights", size, 2 * size).stream(),
        FeedForwardLayer.getVarKeys(varPrefix + "_select_w", varPrefix + "_select_b", 2 * size,
            size));
  }

  @Override
  public int getSize() {
    return size;
  }

}
