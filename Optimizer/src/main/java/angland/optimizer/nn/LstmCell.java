package angland.optimizer.nn;

import java.util.Map;
import java.util.stream.Stream;

import angland.optimizer.var.IMatrixValue;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.ScalarValue;


public class LstmCell {

  private FeedForwardLayer<String> retain;

  private static final ScalarValue<String> one = ScalarValue.constant(1);
  private static final ScalarValue<String> minusOne = ScalarValue.constant(1);
  private final int size;
  private FeedForwardLayer<String> modify;
  private FeedForwardLayer<String> select;

  public LstmCell(String varPrefix, int size, Map<IndexedKey<String>, Double> context) {
    this.retain =
        new FeedForwardLayer<>(size, size, ScalarValue::sigmoid, varPrefix + "_retain_w", varPrefix
            + "_retain_b", context);
    this.modify =
        new FeedForwardLayer<>(size, size, ScalarValue::tanh, varPrefix + "_modify_w", varPrefix
            + "_modify_b", context);
    this.select =
        new FeedForwardLayer<>(size, size, ScalarValue::sigmoid, varPrefix + "_select_w", varPrefix
            + "_select_b", context);
    this.size = size;
  }

  public LstmStateTuple<String> apply(LstmStateTuple<String> input) {

    IMatrixValue<String> retainHidden = retain.apply(input.getExposedState());

    IMatrixValue<String> replaceHidden = minusOne.times(retainHidden).transform(s -> s.plus(one));

    IMatrixValue<String> modifier = modify.apply(input.getExposedState());
    IMatrixValue<String> replaceModify = replaceHidden.pointwiseMultiply(modifier);

    IMatrixValue<String> hiddenModified =
        input.getHiddenState().pointwiseMultiply(retainHidden).plus(replaceModify);


    IMatrixValue<String> selector = select.apply(input.getExposedState());

    IMatrixValue<String> cellOutput = selector.pointwiseMultiply(hiddenModified);

    return new LstmStateTuple<>(hiddenModified, cellOutput);
  }

  public int getSize() {
    return size;
  }

  public static Stream<IndexedKey<String>> getKeys(String varPrefix, int size) {
    return Stream.concat(FeedForwardLayer.getVarKeys(varPrefix + "_retain_w", varPrefix
        + "_retain_b", size, size), Stream.concat(
        FeedForwardLayer.getVarKeys(varPrefix + "_modify_w", varPrefix + "_modify_b", size, size),
        FeedForwardLayer.getVarKeys(varPrefix + "_select_w", varPrefix + "_select_b", size, size)));
  }


}
