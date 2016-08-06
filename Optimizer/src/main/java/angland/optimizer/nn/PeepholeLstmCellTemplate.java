package angland.optimizer.nn;

import java.util.stream.Stream;

import angland.optimizer.var.Context;
import angland.optimizer.var.IndexedKey;

public class PeepholeLstmCellTemplate implements RnnCellTemplate {

  private final String varPrefix;
  private final int size;
  private final double gradientClipThreshold;
  private final boolean isConstant;

  public PeepholeLstmCellTemplate(String varPrefix, int size, double gradientClipThreshold,
      boolean isConstant) {
    super();
    this.varPrefix = varPrefix;
    this.size = size;
    this.gradientClipThreshold = gradientClipThreshold;
    this.isConstant = isConstant;
  }

  @Override
  public RnnCell<String> create(Context<String> context) {
    return new PeepholeLstmCell(varPrefix, size, context, gradientClipThreshold, isConstant);
  }

  @Override
  public Stream<IndexedKey<String>> getKeys() {
    return Stream.concat(FeedForwardLayer.getVarKeys(varPrefix + "_retain_w", varPrefix
        + "_retain_b", 2 * size, size), Stream.concat(FeedForwardLayer.getVarKeys(varPrefix
        + "_modify_w", varPrefix + "_modify_b", 2 * size, size), FeedForwardLayer.getVarKeys(
        varPrefix + "_select_w", varPrefix + "_select_b", 2 * size, size)));
  }

  @Override
  public int getSize() {
    return size;
  }

}
