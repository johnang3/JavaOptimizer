package angland.optimizer.nn;

import java.util.stream.Stream;

import angland.optimizer.var.Context;
import angland.optimizer.var.IndexedKey;

public class LstmCellTemplate implements RnnCellTemplate {

  private final String varPrefix;
  private final int size;
  private final double gradientClipThreshold;
  private final boolean isConstant;

  public LstmCellTemplate(String varPrefix, int size, double gradientClipThreshold,
      boolean isConstant) {
    super();
    this.varPrefix = varPrefix;
    this.size = size;
    this.gradientClipThreshold = gradientClipThreshold;
    this.isConstant = isConstant;
  }

  @Override
  public RnnCell<String> create(Context<String> context) {
    return new LstmCell<>(varPrefix, size, context, gradientClipThreshold, isConstant);
  }

  @Override
  public Stream<IndexedKey<String>> getKeys() {
    return Stream.concat(FeedForwardLayer.getVarKeys(varPrefix + "_retain_w", varPrefix
        + "_retain_b", size, size), Stream.concat(
        FeedForwardLayer.getVarKeys(varPrefix + "_modify_w", varPrefix + "_modify_b", size, size),
        FeedForwardLayer.getVarKeys(varPrefix + "_select_w", varPrefix + "_select_b", size, size)));
  }

  @Override
  public int getSize() {
    return size;
  }

}
