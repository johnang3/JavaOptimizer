package angland.optimizer.demos;

import static angland.optimizer.demos.DemoConstants.vocabSize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import angland.optimizer.ngram.NGramTrainer;

public class LstmDemoTrainer {

  public static void main(String[] args) throws IOException {
    train(args[0], args[1], args[2]);
  }

  public static void train(String vocabFile, String trainDir, String contextFile)
      throws IOException {
    int numThreads = 4;
    System.out.println("Initializing trainer.  Numthreads: " + numThreads);
    int samples = 100;
    int batchSize = 50;
    int saveInterval = 4;
    double gradientMultiplier = .5;
    List<String> vocabTokens = new ArrayList<>();
    vocabTokens.add("<unk>");
    try (FileReader fr = new FileReader(vocabFile); BufferedReader br = new BufferedReader(fr)) {
      String line = null;
      while ((line = br.readLine()) != null && vocabTokens.size() < vocabSize) {
        vocabTokens.add(line.split(" ")[0]);
      }
    }
    System.out.println("Vocabulary size " + vocabTokens.size());
    TokenBiMap tbm = new TokenBiMap(vocabTokens, "<unk>");


    List<List<Integer>> trainSentences = new ArrayList<>();
    List<File> filesInFolder =
        Files.walk(Paths.get(trainDir)).filter(Files::isRegularFile).map(Path::toFile)
            .collect(Collectors.toList());

    System.out.println("Loading train data.");
    for (File file : filesInFolder) {
      Files.lines(Paths.get(file.getAbsolutePath())).forEach(line -> {
        List<String> tokens = TokenBiMap.tokenize(line);
        List<Integer> tokenInts = new ArrayList<>();
        for (String token : tokens) {
          int tokenInt = tbm.getIdx(token);
          tokenInts.add(tokenInt);
        }
        if (tokenInts.size() > 1) {
          trainSentences.add(tokenInts);
        }
      });
    }
    System.out.println("Done loading train data.");
    System.out.println("Train sequences: " + trainSentences.size());

    ExecutorService es = null;
    try {
      es = Executors.newFixedThreadPool(numThreads);
      NGramTrainer.train(es, trainSentences, new File(contextFile), vocabSize,
          DemoConstants.getTemplate(false), batchSize, saveInterval, gradientMultiplier, samples);
    } finally {
      if (es != null) {
        es.shutdown();
      }
    }
  }


}
