package angland.optimizer.saver;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import angland.optimizer.var.IndexedKey;

public class StringContext {

  public static Map<IndexedKey<String>, Double> loadContext(String file) {
    return loadContext(new File(file));
  }

  public static Map<IndexedKey<String>, Double> loadContext(File file) {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    try {
      Files.lines(Paths.get(file.getAbsolutePath())).forEach(
          line -> {
            String[] split = line.split(" ");
            String key = split[0];
            List<Integer> indices = new ArrayList<>();
            for (int i = 1; i < split.length - 1; ++i) {
              indices.add(Integer.parseInt(split[i]));
            }
            context.put(new IndexedKey<String>(key, indices),
                Double.parseDouble(split[split.length - 1]));
          });
    } catch (NumberFormatException | IOException e) {
      throw new RuntimeException(e);
    }
    return context;
  }

  public static void saveContext(Map<IndexedKey<String>, Double> context, String file) {
    saveContext(context, new File(file));
  }

  public static void saveContext(Map<IndexedKey<String>, Double> context, File file) {
    try (FileWriter fw = new FileWriter(file); PrintWriter pw = new PrintWriter(fw);) {
      context.forEach((k, v) -> {
        StringBuilder sb = new StringBuilder();
        sb.append(k.getVarKey() + " ");
        for (int i : k.getIndices()) {
          sb.append(i + " ");
        }
        sb.append(v);
        pw.println(sb.toString());
      });
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }


}
