pytagi.nn.slstm
===============

.. py:module:: pytagi.nn.slstm


Classes
-------

.. autoapisummary::

   pytagi.nn.slstm.SLSTM


Module Contents
---------------

.. py:class:: SLSTM(input_size: int, output_size: int, seq_len: int, bias: bool = True, gain_weight: float = 1.0, gain_bias: float = 1.0, init_method: str = 'He')

   Bases: :py:obj:`pytagi.nn.base_layer.BaseLayer`


   Smoothing LSTM layer


   .. py:method:: get_layer_info() -> str


   .. py:method:: get_layer_name() -> str


   .. py:method:: init_weight_bias()


