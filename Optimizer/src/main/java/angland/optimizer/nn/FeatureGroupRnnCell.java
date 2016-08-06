package angland.optimizer.nn;

import java.util.List;

import angland.optimizer.var.matrix.IMatrixValue;
import angland.optimizer.var.scalar.IScalarValue;

public class FeatureGroupRnnCell implements RnnCell<String> {

  private final int size;
  private final List<RnnCell<String>> delegates;
  private final List<List<Integer>> selection;

  public FeatureGroupRnnCell(int size, List<RnnCell<String>> delegates, List<List<Integer>> selection) {
    super();
    this.size = size;
    this.delegates = delegates;
    this.selection = selection;
  }

  @Override
  public RnnStateTuple<String> apply(RnnStateTuple<String> input) {
    IMatrixValue<String> hidden = IMatrixValue.repeat(IScalarValue.constant(0), 0, 1);
    IMatrixValue<String> exposed = IMatrixValue.repeat(IScalarValue.constant(0), 0, 1);
    for (int i = 0; i < delegates.size(); ++i) {
      List<Integer> selectedIndices = selection.get(i);
      IMatrixValue<String> delegatedHiddenInput = input.getHiddenState().getRows(selectedIndices);
      IMatrixValue<String> delegatedExposedInput = input.getExposedState().getRows(selectedIndices);
      RnnStateTuple<String> delegateOutput =
          delegates.get(i).apply(new RnnStateTuple<>(delegatedHiddenInput, delegatedExposedInput));
      hidden = hidden.vCat(delegateOutput.getHiddenState());
      exposed = exposed.vCat(delegateOutput.getExposedState());
    }
    return new RnnStateTuple<>(hidden, exposed);
  }

  @Override
  public int getSize() {
    return size;
  }

}
