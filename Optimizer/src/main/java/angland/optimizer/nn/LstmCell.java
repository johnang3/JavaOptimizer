package angland.optimizer.nn;

import angland.optimizer.var.Context;
import angland.optimizer.var.matrix.IMatrixValue;
import angland.optimizer.var.scalar.IScalarValue;


public class LstmCell<VarKey> implements RnnCell<String> {

  private FeedForwardLayer<String> retain;

  private static final IScalarValue<String> one = IScalarValue.constant(1);
  private static final IScalarValue<String> minusOne = IScalarValue.constant(1);
  private final int size;
  private FeedForwardLayer<String> modify;
  private FeedForwardLayer<String> select;
  private final double gradientClipThreshold;

  public LstmCell(String varPrefix, int size, Context<String> context,
      double gradientClipThreshold, boolean constant) {
    this.retain =
        new FeedForwardLayer<>(size, size, v -> v.sigmoid().clipGradient(gradientClipThreshold),
            varPrefix + "_retain_w", varPrefix + "_retain_b", context, constant);
    this.modify =
        new FeedForwardLayer<>(size, size, v -> v.tanh().clipGradient(gradientClipThreshold),
            varPrefix + "_modify_w", varPrefix + "_modify_b", context, constant);
    this.select =
        new FeedForwardLayer<>(size, size, v -> v.sigmoid().clipGradient(gradientClipThreshold),
            varPrefix + "_select_w", varPrefix + "_select_b", context, constant);
    this.gradientClipThreshold = gradientClipThreshold;
    this.size = size;
  }



  public double getGradientClipThreshold() {
    return gradientClipThreshold;
  }



  public RnnStateTuple<String> apply(RnnStateTuple<String> input) {

    IMatrixValue<String> retainHidden = retain.apply(input.getExposedState());

    IMatrixValue<String> replaceHidden = minusOne.times(retainHidden).transform(s -> s.plus(one));

    IMatrixValue<String> modifier = modify.apply(input.getExposedState());
    IMatrixValue<String> replaceModify = replaceHidden.pointwiseMultiply(modifier);

    IMatrixValue<String> hiddenModified =
        input.getHiddenState().pointwiseMultiply(retainHidden).plus(replaceModify);


    IMatrixValue<String> selector = select.apply(input.getExposedState());

    IMatrixValue<String> cellOutput = selector.pointwiseMultiply(hiddenModified);

    return new RnnStateTuple<>(hiddenModified.transform(IScalarValue::cache),
        cellOutput.transform(x -> x.clipGradient(gradientClipThreshold)));
  }

  public int getSize() {
    return size;
  }



}
