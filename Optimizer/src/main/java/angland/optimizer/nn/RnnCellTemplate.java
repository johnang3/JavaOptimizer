package angland.optimizer.nn;

import java.util.Map;
import java.util.stream.Stream;

import angland.optimizer.var.IndexedKey;

public interface RnnCellTemplate {

  public RnnCell<String> create(Map<IndexedKey<String>, Double> context);

  public Stream<IndexedKey<String>> getKeys();

  public int getSize();

}
