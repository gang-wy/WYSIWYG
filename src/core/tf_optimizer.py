"""
Transfer Function Optimizer (Nelder-Mead)

Scipy 기반 Nelder-Mead 최적화.
추후 diff_optimizer.py로 통합 예정.
"""

import numpy as np
from scipy.optimize import minimize


class TFOptimizer:
    def __init__(self, analyzer_data, current_tf_nodes):
        """
        Args:
            analyzer_data: FeatureAnalyzer 결과
            current_tf_nodes: 현재 적용 중인 TF 노드 리스트 (초기 상태)
        """
        self.data = analyzer_data
        self.initial_nodes = current_tf_nodes
        self.tents_base_info = self._extract_all_tents(self.initial_nodes)

        self.x0 = [float(tent['peak']) for tent in self.tents_base_info]
        self.external_loss_callback = None

    def _extract_all_tents(self, nodes):
        """
        [Intensity, R, G, B, A] 노드 리스트에서 개별 Tent 구조를 분리 및 추출
        결과: [{'mu': 중심, 'r':.., 'g':.., 'b':.., 'left':.., 'right':.., 'peak':..}, ...]
        """
        tents = []
        n = len(nodes)
        i = 0
        
        while i < n:
            if nodes[i][4] > 0:
                peak_node = nodes[i]
                left_node = nodes[i-1] if i > 0 else nodes[i]
                right_node = nodes[i+1] if i < n-1 else nodes[i]
                
                mu = peak_node[0]
                left_dist = left_node[0]
                right_dist = right_node[0]
                op_peak = peak_node[4]
                
                tents.append({
                    'mu': mu,
                    'r': peak_node[1], 
                    'g': peak_node[2], 
                    'b': peak_node[3],
                    'left': left_dist,
                    'right': right_dist,
                    'peak': op_peak
                })
                i += 2
            else:
                i += 1
        
        print(f"📊 Detected {len(tents)} Tents from TF nodes.")
        return tents

    def optimize(self, ftol=1e-3, maxiter=50, loss_callback=None):
        """
        Nelder-Mead 최적화 실행
        
        Args:
            ftol: 수렴 허용 오차
            maxiter: 최대 반복 횟수
            loss_callback: 외부 loss 계산 함수 (params -> float)
        """
        self.external_loss_callback = loss_callback
        
        res = minimize(
            self._loss_wrapper,
            self.x0,
            method='Nelder-Mead',
            options={'xatol': ftol, 'maxiter': maxiter}
        )
        return self._apply_all_tents_to_nodes(res.x)

    def _loss_wrapper(self, params):
        """Minimizer가 호출하는 함수"""
        if self.external_loss_callback:
            return self.external_loss_callback(params)
        else:
            return np.random.random()

    def _apply_all_tents_to_nodes(self, optimized_params):
        """최적화된 파라미터 리스트를 다시 GUI용 [I, R, G, B, A] 노드 리스트로 변환"""
        new_nodes = []
        
        for i, tent in enumerate(self.tents_base_info):
            op_peak = max(0.0, min(1.0, optimized_params[i]))
            
            lft = tent['left']
            rgt = tent['right']
            mu = tent['mu']
            r, g, b = tent['r'], tent['g'], tent['b']
            
            new_nodes.append([lft, 0.0, 0.0, 0.0, 0.0])
            new_nodes.append([mu, r, g, b, op_peak])
            new_nodes.append([rgt, 0.0, 0.0, 0.0, 0.0])
            
        return sorted(new_nodes, key=lambda x: x[0])