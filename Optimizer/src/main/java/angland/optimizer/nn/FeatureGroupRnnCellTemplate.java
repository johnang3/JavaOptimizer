package angland.optimizer.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import angland.optimizer.var.Context;
import angland.optimizer.var.IndexedKey;

public class FeatureGroupRnnCellTemplate implements RnnCellTemplate {

  private final List<RnnCellTemplate> templates;
  private final List<List<Integer>> selection;
  private final int size;

  public static FeatureGroupRnnCellTemplate simple(String prefix, int size, int factor,
      double gradientClipThreshold, boolean isConstant, boolean usePeephole) {
    if (size % factor != 0) {
      throw new IllegalArgumentException("size mod factor must be zero.");
    }
    List<List<Integer>> selection = new ArrayList<>();
    List<RnnCellTemplate> templates = new ArrayList<>();
    for (int i = 0; i < factor; ++i) {
      List<Integer> subSelection = new ArrayList<>();
      for (int j = 0; j < size / factor; ++j) {
        subSelection.add(j + i * (size / factor));
      }
      templates.add(usePeephole ? new PeepholeLstmCellTemplate(prefix + "_" + i, size / factor,
          gradientClipThreshold, isConstant) : new LstmCellTemplate(prefix + "_" + i,
          size / factor, gradientClipThreshold, isConstant));
      selection.add(subSelection);
    }
    return new FeatureGroupRnnCellTemplate(templates, selection, size);
  }

  public FeatureGroupRnnCellTemplate(List<RnnCellTemplate> templates,
      List<List<Integer>> selection, int size) {
    super();
    this.templates = templates;
    this.selection = selection;
    this.size = size;
  }

  @Override
  public RnnCell<String> create(Context<String> context) {
    List<RnnCell<String>> cells =
        templates.stream().map(t -> t.create(context)).collect(Collectors.toList());
    return new FeatureGroupRnnCell(size, cells, selection);
  }

  @Override
  public Stream<IndexedKey<String>> getKeys() {
    return templates.stream().flatMap(RnnCellTemplate::getKeys);
  }

  @Override
  public int getSize() {
    return size;
  }

}
