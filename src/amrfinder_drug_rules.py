"""
Drug-specific prediction rules for AMRFinder output
"""
import pandas as pd
import re


class AMRFinderPredictor:
    """Predict resistance using drug-specific mechanistic rules"""

    def __init__(self):
        self.rules = {
            'ciprofloxacin': self._predict_cipro,
            'levofloxacin': self._predict_cipro,
            'tetracycline': self._predict_tetracycline,
            'doxycycline': self._predict_tetracycline,
            'minocycline': self._predict_tetracycline,
            'trimethoprim/sulfamethoxazole': self._predict_tmp_smx,
            'imipenem': self._predict_carbapenem,
            'meropenem': self._predict_carbapenem,
            'ertapenem': self._predict_carbapenem,
            'cefepime': self._predict_cephalosporin,
            'ceftazidime': self._predict_cephalosporin,
            'ceftriaxone': self._predict_cephalosporin,
            'cefotaxime': self._predict_cephalosporin,
            'gentamicin': self._predict_aminoglycoside,
            'tobramycin': self._predict_aminoglycoside,
            'amikacin': self._predict_aminoglycoside,
        }

    def _predict_cipro(self, genes, mutations):
        """Fluoroquinolone resistance"""
        # Check for mutations (POINT type)
        for mut in mutations:
            symbol = mut.get('Element symbol', '')
            if symbol.lower().startswith('gyra') or symbol.lower().startswith('parc'):
                return 'R'

        # Check for qnr genes
        for gene_row in genes:
            symbol = gene_row.get('Element symbol', '').lower()
            if 'qnr' in symbol or 'aac(6\')-ib-cr' in symbol:
                return 'R'

        return 'S'

    def _predict_tetracycline(self, genes, mutations):
        """Tetracycline resistance"""
        for gene_row in genes:
            symbol = gene_row.get('Element symbol', '').lower()
            if re.match(r'tet\([a-z]\)', symbol):
                return 'R'
            if symbol.startswith('tet') and 'tetr' not in symbol:
                return 'R'

        return 'S'

    def _predict_tmp_smx(self, genes, mutations):
        """TMP/SMX resistance - needs BOTH dfr and sul"""
        has_dfr = False
        has_sul = False

        for gene_row in genes:
            symbol = gene_row.get('Element symbol', '').lower()
            if symbol.startswith('dfr'):
                has_dfr = True
            if symbol.startswith('sul'):
                has_sul = True

        if has_dfr and has_sul:
            return 'R'
        elif has_dfr or has_sul:
            return 'I'
        else:
            return 'S'

    def _predict_carbapenem(self, genes, mutations):
        """Carbapenem resistance"""
        carbapenemases = ['kpc', 'ndm', 'oxa-48', 'imp', 'vim', 'ges']

        for gene_row in genes:
            symbol = gene_row.get('Element symbol', '').lower()
            for carb in carbapenemases:
                if carb in symbol:
                    return 'R'

        return 'S'

    def _predict_cephalosporin(self, genes, mutations):
        """Cephalosporin resistance"""
        for gene_row in genes:
            symbol = gene_row.get('Element symbol', '').lower()
            subclass = str(gene_row.get('Subclass', '')).lower()

            if 'ctx-m' in symbol:
                return 'R'
            if 'esbl' in subclass:
                return 'R'

            if symbol.startswith('shv') or symbol.startswith('tem'):
                if 'esbl' in str(gene_row.get('Class', '')).lower():
                    return 'R'

        return 'S'

    def _predict_aminoglycoside(self, genes, mutations):
        """Aminoglycoside resistance"""
        for gene_row in genes:
            symbol = gene_row.get('Element symbol', '').lower()

            if symbol.startswith('aac') or symbol.startswith('aph') or symbol.startswith('ant'):
                return 'R'

            if symbol.startswith('arm') or symbol.startswith('rmt'):
                return 'R'

        return 'S'

    def predict_genome(self, amrfinder_df, antibiotic):
        """Predict resistance for one genome"""
        antibiotic = antibiotic.lower().strip()

        if antibiotic not in self.rules:
            return self._fallback_prediction(amrfinder_df, antibiotic)

        # Split by Type column (AMR vs POINT)
        genes = amrfinder_df[amrfinder_df['Type'] == 'AMR'].to_dict('records')
        mutations = amrfinder_df[amrfinder_df['Type']
                                 == 'POINT'].to_dict('records')

        return self.rules[antibiotic](genes, mutations)

    def _fallback_prediction(self, amrfinder_df, antibiotic):
        """Fallback for unknown drugs"""
        if len(amrfinder_df) == 0:
            return 'S'
        return 'S'
