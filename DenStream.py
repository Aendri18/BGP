import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from collections import defaultdict
import time


class MicroCluster:
    """Represents a micro-cluster in DenStream algorithm"""
    
    def __init__(self, center, creation_time, lambd, mu=1):
        self.center = np.array(center)
        self.creation_time = creation_time
        self.last_update_time = creation_time
        self.lambd = lambd  # fading factor parameter
        self.mu = mu  # minimum points threshold
        self.weight = 1.0
        self.sum_x = np.array(center)
        self.sum_x2 = np.array(center) ** 2
        self.n = 1
        
    def insert_point(self, point, timestamp):
        """Insert a new point into the micro-cluster"""
        fade_factor = 2 ** (-self.lambd * (timestamp - self.last_update_time))
        
        # Apply fading to existing statistics
        self.weight *= fade_factor
        self.sum_x *= fade_factor
        self.sum_x2 *= fade_factor
        
        # Add new point
        self.weight += 1
        self.sum_x += np.array(point)
        self.sum_x2 += np.array(point) ** 2
        self.n += 1
        
        # Update center
        self.center = self.sum_x / self.weight
        self.last_update_time = timestamp
        
    def get_weight(self, current_time):
        """Get current weight considering fading"""
        fade_factor = 2 ** (-self.lambd * (current_time - self.last_update_time))
        return self.weight * fade_factor
        
    def get_radius(self):
        """Calculate cluster radius"""
        if self.weight <= 0:
            return float('inf')
        
        variance = (self.sum_x2 / self.weight) - (self.center ** 2)
        variance = np.maximum(variance, 0)  # Ensure non-negative
        return np.sqrt(np.sum(variance))


class DenStream:
    
    
    def __init__(self, epsilon=0.5, mu=3, beta=0.2, lambd=0.25, init_period=100):
        self.epsilon = epsilon 
        self.mu = mu  
        self.beta = beta  
        self.lambd = lambd 
        self.init_period = init_period
        
        self.p_micro_clusters = []  # potential core micro-clusters
        self.o_micro_clusters = []  # outlier micro-clusters
        self.timestamp = 0
        self.initialized = False
        
    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def _find_closest_cluster(self, point, clusters):
        """Find the closest micro-cluster to a point"""
        if not clusters:
            return None, float('inf')
            
        min_dist = float('inf')
        closest_cluster = None
        
        for cluster in clusters:
            dist = self._distance(point, cluster.center)
            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster
                
        return closest_cluster, min_dist
    
    def _merge_clusters(self, point, timestamp):
        
       
        closest_p, dist_p = self._find_closest_cluster(point, self.p_micro_clusters)
        
        if closest_p and dist_p <= self.epsilon:
            closest_p.insert_point(point, timestamp)
            return True, False  # potential microcluster 
            
        
        closest_o, dist_o = self._find_closest_cluster(point, self.o_micro_clusters)
        
        if closest_o and dist_o <= self.epsilon:
            closest_o.insert_point(point, timestamp)
            
            
            if closest_o.get_weight(timestamp) >= self.mu:
                self.p_micro_clusters.append(closest_o)
                self.o_micro_clusters.remove(closest_o)
                
            return True, False  # if outlier can be promoted to potential
            
        
        new_cluster = MicroCluster(point, timestamp, self.lambd, self.mu)
        self.o_micro_clusters.append(new_cluster)
        
        return False, True  # created new outlier micro-cluster
    
    def _cleanup_clusters(self, timestamp):
        """Remove weak micro-clusters"""
        threshold = self.beta * self.mu
        
       
        self.o_micro_clusters = [
            cluster for cluster in self.o_micro_clusters 
            if cluster.get_weight(timestamp) >= threshold
        ]
        
       
        self.p_micro_clusters = [
            cluster for cluster in self.p_micro_clusters 
            if cluster.get_weight(timestamp) >= self.mu
        ]
    
    def fit_predict_stream(self, X):
        
        predictions = []
        
        for i, point in enumerate(X):
            self.timestamp = i
            
            
            if not self.initialized and i < self.init_period:
                merged, is_anomaly = self._merge_clusters(point, self.timestamp)
                predictions.append(0)  # No anomalies during initialization
                
                if i == self.init_period - 1:
                    self.initialized = True
                    
            else:
                
                merged, is_anomaly = self._merge_clusters(point, self.timestamp)
                
                #  cleanup
                if i % 100 == 0:
                    self._cleanup_clusters(self.timestamp)
                
                predictions.append(-1 if is_anomaly else 1)
                
        return np.array(predictions)

def analyze_bgp_with_denstream():
    """Analyze BGP data using DenStream following the paper's approach"""
    
    
    print("Loading BGP data...")
    df = pd.read_csv("/content/bgpclear_no_traffic.csv")
    df = df.dropna(subset=['state'])
    
    mapping = {"im-state-up": 0, "im-state-down": 1, "im-state-admindown": 2}
    df["label"] = df["state"].map(mapping)
    
    x = df.select_dtypes(include=["int64", "float64"]).fillna(0)
    y = df["label"]
    
   
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    print("\nClass distribution:")
    class_dist = y.value_counts(normalize=True) * 100
    for cls, pct in class_dist.items():
        name = {0: 'up', 1: 'down', 2: 'admindown'}[cls]
        print(f"{name}: {pct:.1f}%")
    
    anomaly_rate = class_dist[1] + class_dist[2] if 1 in class_dist and 2 in class_dist else (
        class_dist[1] if 1 in class_dist else class_dist[2] if 2 in class_dist else 0
    )
    print(f"Total anomaly rate: {anomaly_rate:.1f}%")
    
    
    outlier_summary = {}
    
    for cls in [0, 1, 2]:
        x_class = x_scaled[y == cls]
        y_class = y[y == cls]
        
        if len(x_class) > 100:
            print(f"\n--- Processing class {cls} ({['up', 'down', 'admindown'][cls]}) ---")
            
           
            denstream = DenStream(
                epsilon=0.5,        
                mu=3,              
                beta=0.2,           
                lambd=0.001,        
                init_period=min(100, len(x_class) // 10)
            )
            
            
            print("Processing data stream...")
            preds = denstream.fit_predict_stream(x_class)
            
            # Count anomalies 
            init_period = denstream.init_period
            valid_preds = preds[init_period:]
            outliers = (valid_preds == -1).sum()
            total_valid = len(valid_preds)
            
            outlier_summary[cls] = {
                "total": len(x_class),
                "valid_samples": total_valid,
                "outliers": outliers,
                "outlier_rate": outliers / total_valid if total_valid > 0 else 0,
                "core_clusters": len(denstream.p_micro_clusters),
                "outlier_clusters": len(denstream.o_micro_clusters)
            }
            
            print(f"Core micro-clusters: {len(denstream.p_micro_clusters)}")
            print(f"Outlier micro-clusters: {len(denstream.o_micro_clusters)}")
    
   
    print("\n" + "="*50)
    print("DENSTREAM ANOMALY DETECTION SUMMARY")
    print("="*50)
    
    for cls, stats in outlier_summary.items():
        name = {0: 'up', 1: 'down', 2: 'admindown'}[cls]
        print(f"\nClass: {name}")
        print(f"Total samples: {stats['total']}")
        print(f"Valid samples (after init): {stats['valid_samples']}")
        print(f"Detected outliers: {stats['outliers']}")
        print(f"Outlier rate: {stats['outlier_rate']:.4f} ({stats['outlier_rate']*100:.2f}%)")
       
    
    return outlier_summary, denstream
