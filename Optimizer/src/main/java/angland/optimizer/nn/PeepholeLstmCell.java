package angland.optimizer.nn;

import angland.optimizer.var.Context;
import angland.optimizer.var.matrix.IMatrixValue;
import angland.optimizer.var.scalar.IScalarValue;

public class PeepholeLstmCell implements RnnCell<String> {

  private final FeedForwardLayer<String> retainLayer;
  private final FeedForwardLayer<String> modifyLayer;
  private final FeedForwardLayer<String> selectLayer;
  private final double gradientClipThreshold;
  private final int size;


  public PeepholeLstmCell(String varPrefix, int size, Context<String> context,
      double gradientClipThreshold, boolean constant) {
    this.retainLayer =
        new FeedForwardLayer<>(2 * size, size, v -> v.sigmoid().clipGradient(gradientClipThreshold)
            .cache(5 * size ), varPrefix + "_retain_w", varPrefix + "_retain_b", context,
            constant);
    this.modifyLayer =
        new FeedForwardLayer<>(2 * size, size, v -> v.tanh().clipGradient(gradientClipThreshold)
            .cache(5 * size ), varPrefix + "_modify_w", varPrefix + "_modify_b", context,
            constant);
    this.selectLayer =
        new FeedForwardLayer<>(2 * size, size, v -> v.sigmoid().clipGradient(gradientClipThreshold)
            .cache(5 * size ), varPrefix + "_select_w", varPrefix + "_select_b", context,
            constant);
    this.gradientClipThreshold = gradientClipThreshold;
    this.size = size;
  }



  public double getGradientClipThreshold() {
    return gradientClipThreshold;
  }

  @Override
  public RnnStateTuple<String> apply(RnnStateTuple<String> input) {

    IMatrixValue<String> combinedInputs = input.getHiddenState().vCat(input.getExposedState());

    IMatrixValue<String> retainHidden = retainLayer.apply(combinedInputs);

    IScalarValue<String> one = IScalarValue.constant(1);
    IMatrixValue<String> replaceHidden = retainHidden.transform(x -> one.minus(x));

    IMatrixValue<String> modifier = modifyLayer.apply(combinedInputs);
    IMatrixValue<String> replaceModify = replaceHidden.pointwiseMultiply(modifier);

    IMatrixValue<String> hiddenModified =
        input.getHiddenState().pointwiseMultiply(retainHidden).plus(replaceModify)
            .transform(x -> x.discardBeyond(5 * size));

    IMatrixValue<String> combinedUpdated = hiddenModified.vCat(input.getExposedState());

    IMatrixValue<String> selector = selectLayer.apply(combinedUpdated);
    IMatrixValue<String> selectedOutput =
        selector.pointwiseMultiply(hiddenModified).transform(
            x -> x.clipGradient(gradientClipThreshold));
    return new RnnStateTuple<>(hiddenModified, selectedOutput.transform(x -> x
        .discardBeyond(5 * size)));
  }

  @Override
  public int getSize() {
    return size;
  }
}
