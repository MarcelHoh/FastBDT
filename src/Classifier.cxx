/*
 * Thomas Keck 2017
 *
 * Simplified sklearn interface
 */


#include "Classifier.h"
#include <iostream>
#include <unordered_set>
#include <numeric>

namespace FastBDT {

  void Classifier::fit(const std::vector<std::vector<float>> &X, const std::vector<unsigned int> &y, const std::vector<Weight> &w) {

    if(static_cast<int>(X.size()) - static_cast<int>(m_numberOfFlatnessFeatures) <= 0) {
      throw std::runtime_error("FastBDT requires at least one feature");
    }
    m_numberOfFeatures = X.size() - m_numberOfFlatnessFeatures ;

    if(m_binning.size() == 0) {
      for(unsigned int i = 0; i < X.size(); ++i)
        m_binning.push_back(8);
    }

    if(m_numberOfFeatures + m_numberOfFlatnessFeatures != m_binning.size()) {
      throw std::runtime_error("Number of features must be equal to the number of provided binnings");
    }
    
    if(m_purityTransformation.size() == 0) {
      for(unsigned int i = 0; i < m_binning.size() - m_numberOfFlatnessFeatures; ++i)
        m_purityTransformation.push_back(false);
    }

    for(auto p : m_purityTransformation)
      if(p)
        m_can_use_fast_forest = false;
    
    if(m_numberOfFeatures != m_purityTransformation.size()) {
      throw std::runtime_error("Number of ordinary features must be equal to the number of provided purityTransformation flags.");
    }

    unsigned int numberOfEvents = X[0].size();
    if(numberOfEvents == 0) {
      throw std::runtime_error("FastBDT requires at least one event");
    }

    if(numberOfEvents != y.size()) {
      throw std::runtime_error("Number of data-points X doesn't match the numbers of labels y");
    }
    
    if(numberOfEvents != w.size()) {
      throw std::runtime_error("Number of data-points X doesn't match the numbers of weights w");
    }


    // derive the number of classes
    std::unordered_set<unsigned int> yHashMap;
    std::map<unsigned int, unsigned int> classLabelToIndex;

    // O(n)
    m_nClasses = 0; // use as counter and to then store the total number of classes
    std::vector<unsigned int> nEventsPerClass = {};
    std::vector<unsigned int> yClass(y.size(), 0);

    for (unsigned int iY = 0; iY < y.size(); ++iY) {
      if (yHashMap.find(y[iY]) == yHashMap.end()) {
        yHashMap.insert(y[iY]);
        classLabelToIndex[y[iY]] = m_nClasses;
        m_classIndexToLabel.push_back(y[iY]);
        nEventsPerClass.push_back(1);
        m_nClasses++;
      }
      else {
        nEventsPerClass[classLabelToIndex[y[iY]]]++;
      }
      yClass[iY] = classLabelToIndex[y[iY]];
    }

    // For two class BDT maintain the behaviour that class with label = 1 is the signal class
    // From here on we assume the signal class has index = 0. If this does not agree with the label we need to update a few things
    if (m_nClasses==2) {
      if (m_classIndexToLabel[0] != 1) {
        if (m_classIndexToLabel[1] == 1) {
          // need to switch index 0 and 1 for everything
          // this is slow. Could be avoided by asking the user to provide how many classes there will be - might also be safer.
          unsigned int nEvents0 = nEventsPerClass[0];
          nEventsPerClass[0] = nEventsPerClass[1];
          nEventsPerClass[1] = nEvents0;

          unsigned int classLabel0 = m_classIndexToLabel[0];
          m_classIndexToLabel[0] = m_classIndexToLabel[1];
          m_classIndexToLabel[1] = classLabel0;

          classLabelToIndex[m_classIndexToLabel[0]] = 0;
          classLabelToIndex[m_classIndexToLabel[1]] = 1;

          // reset the class array
          for (unsigned int iY = 0; iY < y.size(); ++iY) {
                  yClass[iY] = classLabelToIndex[y[iY]];
          }
        }
          // if this is not the case neither class has a 1 label.
          // Keep the labels in the order we found them.
          // The user can check the order of the labels to check what is considered the signal class
      } 
    }

    std::vector<unsigned int> startingIndexPerClass(nEventsPerClass.size()+1, 0);
    std::partial_sum(nEventsPerClass.begin(), nEventsPerClass.end(), startingIndexPerClass.begin()+1, std::plus<unsigned int>());
    startingIndexPerClass.resize(nEventsPerClass.size());

    bool anyPurityTransforms = std::any_of(m_purityTransformation.begin(), m_purityTransformation.end(), [](bool i){return i;});

    std::vector<bool> yBool;

    if (anyPurityTransforms) {
      if (m_nClasses != 2) {
        throw std::runtime_error("Purity transforms are currently only supported for binary classification. Not multiclass.");
      }
      //assumes signal class has index 0;
      yBool.resize(yClass.size());
      for (unsigned int iY = 0; iY < yClass.size(); ++iY) {
        yBool[iY] = yClass[iY] == 0;
      }
    }

    m_numberOfFinalFeatures = m_numberOfFeatures;
    for(unsigned int iFeature = 0; iFeature < m_numberOfFeatures; ++iFeature) {
      auto feature = X[iFeature];
      m_featureBinning.push_back(FeatureBinning<float>(m_binning[iFeature], feature));
      if(m_purityTransformation[iFeature]) {
        m_numberOfFinalFeatures++;
        std::vector<unsigned int> feature(numberOfEvents);
        for(unsigned int iEvent = 0; iEvent < numberOfEvents; ++iEvent) {
          feature[iEvent] = m_featureBinning[iFeature].ValueToBin(X[iFeature][iEvent]);
        }
        m_purityBinning.push_back(PurityTransformation(m_binning[iFeature], feature, w, yBool));
        m_binning.insert(m_binning.begin() + iFeature + 1, m_binning[iFeature]);
      }
    }
    
    for(unsigned int iFeature = 0; iFeature < m_numberOfFlatnessFeatures; ++iFeature) {
      auto feature = X[iFeature + m_numberOfFeatures];
      m_featureBinning.push_back(FeatureBinning<float>(m_binning[iFeature + m_numberOfFinalFeatures], feature));
    }
    
    EventSample eventSample(numberOfEvents, m_nClasses, startingIndexPerClass, m_numberOfFinalFeatures, m_numberOfFlatnessFeatures, m_binning);
    std::vector<unsigned int> bins(m_numberOfFinalFeatures+m_numberOfFlatnessFeatures);

    for(unsigned int iEvent = 0; iEvent < numberOfEvents; ++iEvent) {
      unsigned int bin = 0;
      unsigned int pFeature = 0; 
      for(unsigned int iFeature = 0; iFeature < m_numberOfFeatures; ++iFeature) {
        bins[bin] = m_featureBinning[iFeature].ValueToBin(X[iFeature][iEvent]);
        bin++;
        if(m_purityTransformation[iFeature]) {
          bins[bin] = m_purityBinning[pFeature].BinToPurityBin(bins[bin-1]);
          pFeature++;
          bin++;
        }
      }
      for(unsigned int iFeature = 0; iFeature < m_numberOfFlatnessFeatures; ++iFeature) {
        bins[bin] = m_featureBinning[iFeature + m_numberOfFeatures].ValueToBin(X[iFeature + m_numberOfFeatures][iEvent]);
        bin++;
      }
      eventSample.AddEvent(bins, w[iEvent], yClass[iEvent]);
    }
   
    m_featureBinning.resize(m_numberOfFeatures);

    ForestBuilder df(eventSample, m_nTrees, m_shrinkage, m_subsample, m_depth, m_nClasses, m_sPlot, m_flatnessLoss);
    if(m_can_use_fast_forest) {
        Forest<float> temp_forest( df.GetShrinkage(), df.GetF0(), m_transform2probability, m_nClasses);
        for( auto t : df.GetForest() ) {
           temp_forest.AddTree(removeFeatureBinningTransformationFromTree(t, m_featureBinning));
        }
        m_fast_forest = temp_forest;
    } else {
        Forest<unsigned int> temp_forest(df.GetShrinkage(), df.GetF0(), m_transform2probability, m_nClasses);
        for( auto t : df.GetForest() ) {
           temp_forest.AddTree(t);
        }
        m_binned_forest = temp_forest;
    }

  }

  void Classifier::Print() {

    std::cout << "NTrees " << m_nTrees << std::endl;
    std::cout << "Depth " << m_depth << std::endl;
    std::cout << "NumberOfFeatures " << m_numberOfFeatures << std::endl;

  }
      
  std::vector<float> Classifier::predict(const std::vector<float> &X) const {

      if(m_can_use_fast_forest) {
        if (m_nClasses == 2) {
          return m_fast_forest.Analyse(X);
        } else {
          return m_fast_forest.AnalyseMulticlass(X);
        }
      } else {
        std::vector<unsigned int> bins(m_numberOfFinalFeatures);
        unsigned int bin = 0;
        unsigned int pFeature = 0;
        for(unsigned int iFeature = 0; iFeature < m_numberOfFeatures; ++iFeature) {
          bins[bin] = m_featureBinning[iFeature].ValueToBin(X[iFeature]);
          bin++;
          if(m_purityTransformation[iFeature]) {
              bins[bin] = m_purityBinning[pFeature].BinToPurityBin(bins[bin-1]);
              pFeature++;
              bin++;
          }
        }
        if (m_nClasses == 2) {
          return m_binned_forest.Analyse(bins);
        } else {
          return m_binned_forest.AnalyseMulticlass(bins);
        }
      }
  }
  
  std::map<unsigned int, double> Classifier::GetIndividualVariableRanking(const std::vector<float> &X) const {
    
      std::map<unsigned int, double> ranking;

      if(m_can_use_fast_forest) {
        ranking = m_fast_forest.GetIndividualVariableRanking(X);
      } else {
        std::vector<unsigned int> bins(m_numberOfFinalFeatures);
        unsigned int bin = 0;
        unsigned int pFeature = 0;
        for(unsigned int iFeature = 0; iFeature < m_numberOfFeatures; ++iFeature) {
          bins[bin] = m_featureBinning[iFeature].ValueToBin(X[iFeature]);
          bin++;
          if(m_purityTransformation[iFeature]) {
              bins[bin] = m_purityBinning[pFeature].BinToPurityBin(bins[bin-1]);
              pFeature++;
              bin++;
          }
        }
        ranking = m_binned_forest.GetIndividualVariableRanking(bins);
      }

      return MapRankingToOriginalFeatures(ranking);
  }

  std::map<unsigned int, unsigned int> Classifier::GetFeatureMapping() const {
    
    std::map<unsigned int, unsigned int> transformed2original;
    unsigned int transformedFeature = 0;
    for(unsigned int originalFeature = 0; originalFeature < m_numberOfFeatures; ++originalFeature) {
      transformed2original[transformedFeature] = originalFeature;
      if(m_purityTransformation[originalFeature]) {
        transformedFeature++;
        transformed2original[transformedFeature] = originalFeature;
      }
      transformedFeature++;
    }

    return transformed2original;

  }

  std::map<unsigned int, double> Classifier::MapRankingToOriginalFeatures(std::map<unsigned int, double> ranking) const {
    auto transformed2original = GetFeatureMapping();
    std::map<unsigned int, double> original_ranking;
    for(auto &pair : ranking) {
      if(original_ranking.find(transformed2original[pair.first]) == original_ranking.end())
        original_ranking[transformed2original[pair.first]] = 0;
      original_ranking[transformed2original[pair.first]] += pair.second;
    }
    return original_ranking;
  }


  std::map<unsigned int, double> Classifier::GetVariableRanking() const {
    std::map<unsigned int, double> ranking;
    if (m_can_use_fast_forest)
      ranking = m_fast_forest.GetVariableRanking();
    else
      ranking = m_binned_forest.GetVariableRanking();
    return MapRankingToOriginalFeatures(ranking);
  }


std::ostream& operator<<(std::ostream& stream, const Classifier& classifier) {

    stream << classifier.m_version << std::endl;
    stream << classifier.m_nTrees << std::endl;
    stream << classifier.m_depth << std::endl;
    stream << classifier.m_nClasses << std::endl;
    stream << classifier.m_classIndexToLabel << std::endl;
    stream << classifier.m_binning << std::endl;
    stream << classifier.m_shrinkage << std::endl;
    stream << classifier.m_subsample << std::endl;
    stream << classifier.m_sPlot << std::endl;
    stream << classifier.m_flatnessLoss << std::endl;
    stream << classifier.m_purityTransformation << std::endl;
    stream << classifier.m_transform2probability << std::endl;
    stream << classifier.m_featureBinning << std::endl;
    stream << classifier.m_purityBinning << std::endl;
    stream << classifier.m_numberOfFeatures << std::endl;
    stream << classifier.m_numberOfFinalFeatures << std::endl;
    stream << classifier.m_numberOfFlatnessFeatures << std::endl;
    stream << classifier.m_can_use_fast_forest << std::endl;
    stream << classifier.m_fast_forest << std::endl;
    stream << classifier.m_binned_forest << std::endl;

    return stream;
}

}
