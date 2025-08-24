# Tabs for different draft analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Over/Under Performance", "📊 Draft vs Final Position", "🍀 Lottery Luck", "🎯 Manager Performance", "📈 Draft Value Analysis"])
            
            with tab1:
                st.subheader("Over/Under Performance")
                st.markdown("*Diskrepanz zwischen Draft-Position und finalem Rang*")
                
                # Season filter
                seasons = sorted(draft_analysis_df['Season'].unique(), reverse=True)
                selected_season = st.selectbox("Saison auswählen:", ["Alle Saisons"] + list(seasons))
                if selected_season != "Alle Saisons":
                    filtered_df = draft_analysis_df[draft_analysis_df['Season'] == selected_season]
                else:
                    filtered_df = draft_analysis_df
                
                # Top Over/Underperformers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🚀 Beste Overperformer")
                    st.markdown("*Höchste positive Abweichung (Draft schlechter als Endrang)*")
                    best_over = filtered_df.nlargest(5, 'Over_Under')
                    
                    for i, (_, row) in enumerate(best_over.iterrows()):
                        st.markdown(f"""
                        <div class="favorite-opponent">
                            <h4>#{i+1} {row['Manager']} ({row['Season']})</h4>
                            <p><strong>+{row['Over_Under']}</strong> (Pick {row['Draft_Position']} → Rang {row['Final_Rank']})</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 😰 Größte Underperformer")  
                    st.markdown("*Höchste negative Abweichung (Draft besser als Endrang)*")
                    worst_under = filtered_df.nsmallest(5, 'Over_Under')
                    
                    for i, (_, row) in enumerate(worst_under.iterrows()):
                        st.markdown(f"""
                        <div class="nightmare-opponent">
                            <h4>#{i+1} {row['Manager']} ({row['Season']})</h4>
                            <p><strong>{row['Over_Under']}</strong> (Pick {row['Draft_Position']} → Rang {row['Final_Rank']})</p>
                        </div>
                        """, unsafe_allow_html=True)
