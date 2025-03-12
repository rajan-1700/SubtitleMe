import os
import subprocess
import librosa
from transformers import pipeline
from flask import Flask, request, render_template
import requests
